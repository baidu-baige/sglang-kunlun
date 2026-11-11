# DeepSeek-V4.1-Flash EPD example (SGLang 0.5.21 / Kunlun P800)

Runnable launch scripts for the validated DeepSeek-V4.1-Flash deployment: one
encoder role, one language-only prefill role, one disaggregated decode role,
and `sglang-router` in front.

| Script | Role | Runs on |
| --- | --- | --- |
| `start_encoder.sh` | Eight single-XPU vision workers on ports 8000-8007 | encoder host |
| `start_prefill.sh` | tp8/ep8/cp8 language-only prefill, sends KV over mooncake | prefill host (8 XPUs) |
| `start_decode.sh` | tp8/ep8 decode, low-latency DeepEP + CUDA graphs | decode host (8 XPUs) |
| `start_router.sh` | OpenAI-compatible front end with PD routing | any host with `sglang-router` |

## Prerequisites

- An image built from [`dockerfile/`](../../dockerfile/README.md): community
  SGLang 0.5.21, the Kunlun XPU runtime and the operator wheels.
- Eight Kunlun3 P800 XPUs on the prefill host, eight on the decode host, and
  one XPU per encoder worker.
- RDMA devices for the mooncake KV transfer (auto-probed from `mlx5_*`;
  override with `IB_DEVICE_CANDIDATES`).
- A mooncake metadata server if your deployment uses one
  (`MOONCAKE_META_SERVER_URL` / `MOONCAKE_META_SERVER_PASSWORD`).

## Environment

| Variable | Required | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | all engine roles | DeepSeek-V4.1-Flash W4A8 checkpoint |
| `MODEL_ID` | no | served model name (default `DeepSeek-V4.1-Flash`) |
| `PREFILL_IP` | encoder, router | prefill host IP |
| `DECODE_IP` | router | decode host IP |
| `ENCODER_URLS` | prefill | space-separated `http://<enc-ip>:8000 ... :8007` |
| `PLUGIN_DIR` | no | plugin tree in the image (default `/workspace/sglang_kunlun`) |
| `IB_DEVICE_CANDIDATES` | no | RDMA devices to probe |
| `LIBCUDART_PRELOAD` | no | `libcudart.so.11.0` path when it is not on the loader path |

Per-role kernel switches live in `env.prefill.sh` and `env.decode.sh`; they
mirror the validated SGLang 0.5.21 configuration.

## Run

Start the roles in this order, each on its own host:

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-W4A8
export PREFILL_IP=<prefill-ip>
export DECODE_IP=<decode-ip>
export ENCODER_URLS="http://<encoder-ip>:8000 ... http://<encoder-ip>:8007"

bash start_encoder.sh   # encoder host
bash start_prefill.sh   # prefill host
bash start_decode.sh    # decode host
bash start_router.sh    # router host
```

Wait until the prefill `/health` endpoint and the encoder registration are up
before sending traffic. Confirm the served model name through the router:

```bash
curl http://<router-ip>:8101/v1/models
```

Then send chat requests to the router port.

> The encoder script launches eight workers on one host, one device each. If
> you run them as eight single-XPU containers instead, start one worker per
> container with `ENCODE_WORKERS=1` and `ENCODE_BASE_PORT=8000`.
