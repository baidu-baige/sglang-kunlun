"""DSpark draft-path fixes for Kunlun XPU.

Everything here concerns `exp_noise`, the random input to draft sampling, so
nothing in this module takes effect at temperature 0. Temperature is the switch
because `sampling_params.py:145-149` pins `top_k` to 1 at temperature 0 and to
TOP_K_ALL otherwise, and the sampler turns that into
`denom = where(greedy_mask, 1.0, exp_noise)`: at temperature 0 the noise is
replaced by a constant 1.0 and cannot matter, above 0 it is live.


Background: why the draft sampler needs random numbers
------------------------------------------------------
Sampling at temperature > 0 means drawing from a distribution instead of taking
its maximum. DSpark does that with Gumbel-max:

    token = argmax_i (p_i / E_i),   E_i ~ Exp(1) i.i.d.

which is provably distributed exactly as `p`. `exp_noise` holds those E_i, one
per vocabulary entry, and `SampleStepTokens` divides the probabilities by them.

The trick lives or dies on the *near-zero* draws: a token can only outrank the
highest-probability token if its own E_i happens to be tiny. Noise whose minimum
is bounded away from zero collapses the sampler into greedy decoding, and noise
that is outright garbage lets the same few token ids win at every step, which
reads as a repetition loop.


Background: why a cuda graph breaks an ordinary RNG
--------------------------------------------------
A pseudo-random generator is a state plus a step function. The same state always
yields the same number, so the state must advance on every call. Counter-based
generators such as philox reduce that state to `(seed, offset)`, of which
`offset` is the part that moves.

A cuda graph records a sequence of kernel launches once and then replays it. On
replay no Python and no host code executes, and each kernel sees the arguments
captured earlier. An offset baked in as an immediate at capture time therefore
produces identical numbers on every replay.

On an NVIDIA card the generator state used inside a graph is kept in device memory
and advanced once per replay, so upstream can simply write
`self.exp_noise[:bs].exponential_()` (`dspark_draft_sampler.py:105`) and get a
fresh draw on every replay with no further bookkeeping.

This platform does not refresh that draw. Reading `exp_noise` back after a few
hundred replays gives something that is not Exp(1) at all, and eventually not a
number:

    NVIDIA  mean 1.0010   min 1.9e-6   max 12.8    (Exp(1), as intended)
    Kunlun  mean 29.77
    Kunlun  mean nan                   max inf

`-log(u)/lambda` cannot produce NaN, so `u` itself is uninitialised memory rather
than a stale but valid draw. The same code is unchanged in current upstream and is
correct on an NVIDIA card, so the design is sound and the platform is the
difference. A replacement therefore has to carry its own state.
"""

import logging
from typing import Optional

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

#: State for `_graph_noise`, allocated once before any graph capture and touched
#: only by in-place ops that the graph records. There is one draft sampler per
#: process, so module globals suffice to reach these from the sampler hook.
#:
#: `_CTR` is the generator state: it lives in device memory and is incremented
#: *inside* the graph, so it advances once per markov step per replay with no host
#: involvement and therefore no ordering hazard. This mirrors how a graph-safe RNG
#: keeps its offset on the device rather than baking it into the launch.
_CTR: Optional[torch.Tensor] = None  # generator state, bumped in-graph
_SALT: Optional[torch.Tensor] = None  # _CTR spread across 32 bits
_IDX: Optional[torch.Tensor] = None  # token indices, broadcast over rows
_ROW: Optional[torch.Tensor] = None  # per-row offsets, broadcast over tokens
_HASH: Optional[torch.Tensor] = None  # hash accumulator
_TMP: Optional[torch.Tensor] = None  # scratch for the xor-shift steps
_NOISE: Optional[torch.Tensor] = None  # float32 output handed to the sampler

#: `_M1`/`_M2` and the shift amounts below are the `lowbias32` finalizer, whose
#: job is avalanche: flipping one input bit flips about half the output bits.
#: `_M0` is 2**32/phi, an odd multiplier used to spread consecutive values before
#: mixing -- consecutive counters differ only in their low bits, and XOR-ing
#: those in directly would barely perturb the hash.
_M0 = 0x9E3779B1
_M1 = 0x7FEB352D
_M2 = 0x846CA68B
#: torch has no uint32, so the hash runs in int64 and masks back to 32 bits after
#: every multiply. That emulates unsigned wraparound and keeps values
#: non-negative, which in turn makes `>>` behave as a logical shift.
_U32 = 0xFFFFFFFF
#: float32 has a 24-bit mantissa, so every integer in [0, 2**24] converts
#: exactly. Mapping h -> (h + 1) / (2**24 + 1) keeps u strictly inside (0, 1):
#: u != 0 keeps log(u) finite, u != 1 keeps -log(u) strictly positive, so the
#: sampler's later division can neither divide by zero nor see a negative
#: denominator. The resulting floor of ~6e-8 sits well below the 1/V ~ 8e-6
#: expected minimum over a 129280-entry vocabulary, so the tail that Gumbel-max
#: depends on is not truncated.
_U24 = 0xFFFFFF
_U24_SCALE = 1.0 / float(_U24 + 2)
#: Row decorrelation. Without it, (row 1, token 0) and (row 0, token 1) would
#: hash the same input and share a noise value. The stride is odd, so it loses no
#: low-order information under 32-bit wraparound, and far larger than any
#: vocabulary, so per-row input ranges do not overlap.
_ROW_STRIDE = 0x27D4EB2F

def _graph_noise(bs: int) -> torch.Tensor:
    """Draw Exp(1) noise entirely inside the draft graph.

    Stands in for `exp_noise.exponential_()`, which is not refreshed on replay on
    this platform (see the module docstring). Two stages: hash `(row, token)`
    together with an in-graph counter into a pseudo-random 24-bit integer, then
    turn that integer into an Exp(1) sample by inverse transform sampling.
    """
    # Stage 1a: advance the generator state and derive this step's salt. The
    # increment is a captured kernel, so it really executes on every replay.
    #
    # The salt is XOR-ed into the hash *after* the index has been multiplied,
    # and is never added to the index itself. `hash(idx + ctr)` would make step
    # N+1 the step-N vector shifted by one slot -- the same multiset in a new
    # order, which still gives some token the boost implied by its ~1e-6
    # minimum. XOR-ing post-multiply makes each counter value a genuinely
    # different permutation of idx -> hash.
    _CTR.add_(1)
    torch.mul(_CTR, _M0, out=_SALT)
    _SALT.bitwise_and_(_U32)

    # Stage 1b: hash. `_IDX + _ROW` broadcasts to (bs, vocab), giving each
    # (row, token) a distinct input; the pre-multiply spreads adjacent token
    # indices before the salt goes in; the remaining five steps are lowbias32.
    h = _HASH[:bs]
    t = _TMP[:bs]
    torch.add(_IDX, _ROW[:bs], out=h)
    h.mul_(_M1)
    h.bitwise_and_(_U32)
    h.bitwise_xor_(_SALT)
    torch.bitwise_right_shift(h, 16, out=t)
    h.bitwise_xor_(t)
    h.mul_(_M1)
    h.bitwise_and_(_U32)
    torch.bitwise_right_shift(h, 15, out=t)
    h.bitwise_xor_(t)
    h.mul_(_M2)
    h.bitwise_and_(_U32)
    torch.bitwise_right_shift(h, 16, out=t)
    h.bitwise_xor_(t)
    h.bitwise_and_(_U24)

    # Stage 2: inverse transform sampling. u = (h + 1) / (2**24 + 1) is uniform
    # on (0, 1), and -log(u) is then Exp(1) since P(-log(u) > x) = P(u < e**-x)
    # = e**-x. The int64 -> float32 copy is exact because h < 2**24.
    noise = _NOISE[:bs]
    noise.copy_(h)
    noise.add_(1.0)
    noise.mul_(_U24_SCALE)
    torch.log(noise, out=noise)
    noise.neg_()
    return noise


@plugin_hook(
    target=(
        "sglang.srt.speculative.dspark_components.dspark_draft_sampler."
        "DsparkDraftSampler.__init__"
    ),
    type=HookType.AROUND,
)
def draft_sampler_init_kunlun(original_fn, self, *args, **kwargs):
    """Allocate everything `_graph_noise` needs, before any graph exists.

    Shapes are derived from `self.exp_noise`, the buffer upstream would have
    filled, so that `_graph_noise` can be sliced to any batch size up to `max_bs`.
    Leaving the globals as None (no `exp_noise` attribute) makes
    `sample_step_tokens_torch` fall back to upstream behaviour.
    """
    global _CTR, _SALT, _IDX, _ROW, _HASH, _TMP, _NOISE
    result = original_fn(self, *args, **kwargs)
    noise = getattr(self, "exp_noise", None)
    if noise is None:
        return result
    max_bs, vocab = noise.shape
    dev = noise.device
    _CTR = torch.zeros(1, dtype=torch.int64, device=dev)
    _SALT = torch.zeros(1, dtype=torch.int64, device=dev)
    _IDX = torch.arange(vocab, dtype=torch.int64, device=dev).view(1, vocab)
    _ROW = (
        torch.arange(max_bs, dtype=torch.int64, device=dev).view(max_bs, 1)
        * _ROW_STRIDE
    )
    _HASH = torch.empty((max_bs, vocab), dtype=torch.int64, device=dev)
    _TMP = torch.empty((max_bs, vocab), dtype=torch.int64, device=dev)
    _NOISE = torch.empty_like(noise)
    logger.info(
        "[dspark] in-graph counter-hash draft noise, %s rows x %s vocab, "
        "%.0f MiB pre-allocated",
        max_bs,
        vocab,
        (_HASH.numel() * 8 * 2 + _NOISE.numel() * 4 + _IDX.numel() * 8) / 2**20,
    )
    return result


@plugin_hook(
    target=(
        "sglang.kernels.ops.speculative.dspark.dspark_draft_model."
        "SampleStepTokens.triton"
    ),
    type=HookType.REPLACE,
)
def sample_step_tokens_torch(
    cls,
    *,
    step_logits,
    temperatures,
    greedy_mask,
    exp_noise,
):
    """Draft step sampling: upstream's torch Gumbel-max, on graph-safe noise.

    Noise, which restores the sampled distribution. `_graph_noise` stands in for a
    draw that is not refreshed on replay here. The guard is the contract with
    `draft_sampler_init_kunlun`: substitute only when that hook allocated buffers
    this batch can be sliced out of, otherwise pass upstream's `exp_noise` through
    untouched.
    """
    from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
        sample_step_tokens,
    )

    bs = exp_noise.shape[0]
    if (
        _NOISE is not None
        and _NOISE.shape[1:] == exp_noise.shape[1:]
        and _NOISE.shape[0] >= bs
    ):
        exp_noise = _graph_noise(bs)

    return sample_step_tokens(
        step_logits=step_logits,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )
