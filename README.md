# sglang-kunlun

`sglang-kunlun` 是面向昆仑 XPU 的 SGLang Out-of-Tree 平台插件。它基于 SGLang 0.5.11 之后合入社区的插件机制，通过 `platform + pre-shim + hooks + kernels` 的组合，在尽量不修改 SGLang 主仓代码的前提下，为 Kunlun/XPU 环境提供平台发现、运行时兼容、函数/类替换和 CUDA-only kernel 替换能力。

## 设计目标

- 以 SGLang 官方 OOT 插件机制接入 Kunlun 平台，减少主仓 patch 面。
- 将平台能力、启动前兼容层、Hook 注入和 kernel replacement 分层管理，降低维护和升级成本。
- 保留上游 SGLang 的调用形态，将实际执行路由到 Kunlun/XPU 可用的 `kunlun_ops`、`xspeedgate_ops`、`torch_xmlir` 或 PyTorch fallback。
- 兼容上游 CUDA-only 依赖和 import 时机问题，例如 Triton、`sgl_kernel`、`flashinfer`、`deep_gemm` 等。

## 核心机制

SGLang 主仓通过 Python `setuptools` entry points 自动发现插件。本插件在 `pyproject.toml` 中注册两个入口：

```toml
[project.entry-points."sglang.srt.platforms"]
kunlun = "sglang_kunlun.platform:activate"

[project.entry-points."sglang.srt.plugins"]
kunlun_hooks = "sglang_kunlun.hooks.registry:register_all"

[project.scripts]
sglang-kunlun-launch = "sglang_kunlun.launch:main"
```

整体启动和加载链路如下：

```text
用户启动 / sglang-kunlun-launch
  -> sglang_kunlun.launch
  -> 设置 SGLANG_PLATFORM=kunlun
  -> 执行 _kunlun_pre_shim()
  -> SGLang platform discovery
  -> entry_points: sglang.srt.platforms / kunlun
  -> sglang_kunlun.platform.activate()
  -> KunlunSRTPlatform
  -> SGLang load_plugins()
  -> entry_points: sglang.srt.plugins / kunlun_hooks
  -> sglang_kunlun.hooks.registry.register_all()
  -> kernel_ops.install() + hook modules import
  -> 模型加载 / KV Cache / Attention / MoE / Quant / Speculative / Distributed
```

## 代码结构

```text
sglang-kunlun/
├── pyproject.toml                 # 包元信息与 SGLang 插件入口
├── sitecustomize.py               # Python 启动早期自动 shim
├── README.md                      # 项目说明与运行指引
├── requirements.txt               # 开发与测试依赖
├── test/                          # 插件机制和兼容契约测试
└── sglang_kunlun/
    ├── launch.py                  # 显式启动入口，设置平台并运行 sglang.launch_server
    ├── bootstrap/                 # 启动前 CUDA-like 兼容 shim
    ├── platform/                  # SGLang current_platform 平台插件
    ├── hooks/                     # HookRegistry 函数、类、模块行为替换
    ├── kernels/                   # Triton/JIT/sgl_kernel/可选 CUDA 依赖替换
    └── models/                    # 模型专项适配入口
```

## 分层设计

### 1. Platform 层

`sglang_kunlun.platform.activate()` 是平台发现入口。它会在 `torch_xmlir` 可导入时返回 `KunlunSRTPlatform`，否则返回 `None`，让 SGLang 平台发现逻辑继续回退。

`KunlunSRTPlatform` 继承 `KunlunDeviceMixin` 和 SGLang `SRTPlatform`，负责提供 Kunlun 平台能力：

- `device_name = "kunlun"`
- `device_type = "cuda"`，用于复用上游 CUDA dispatch 路径
- 默认 attention backend 为 `kunlun`
- 默认 `page_size = 128`
- 提供 Kunlun KV Pool、Paged Allocator、Attention Backend 等工厂方法
- 支持 `int8` 量化能力声明

### 2. Bootstrap / pre-shim 层

`sglang_kunlun.bootstrap._kunlun_pre_shim()` 必须在部分 SGLang/CUDA-only 模块 import 前执行，主要用于处理启动早期兼容问题：

- 安装 `sgl_kernel` stub，保证上游 `import sgl_kernel.*` 能正常命中 Kunlun 实现。
- 将 Triton active driver 的 target backend 伪装为 `cuda`，兼容上游 Triton 判断逻辑。
- 为 Kunlun runtime 缺失的 `torch.cuda.memory` 私有 API 提供显式失败的 stub，避免 import 阶段直接崩溃。
- 为 `flashinfer` 可选符号补 stub，避免 CUDA-only 顶层 import 在 Kunlun 环境失败。
- Patch 已加载的 FLA utils，使其在 Kunlun 环境下走兼容设备分支。

`sglang-kunlun-launch` 会显式执行该 shim；包安装后的 `sitecustomize.py` 也用于尽早安装兼容逻辑。

### 3. Hook 层

`sglang_kunlun.hooks.registry.register_all()` 是通用插件入口。SGLang 执行 `load_plugins()` 时会加载该函数，它会：

1. 导入 `deep_geem_hook`、`flashinfer_hook` 和 `kernel_ops`。
2. 执行 `kernel_ops.install()`，安装 Triton/JIT kernel replacement。
3. 按顺序导入各 hook 模块，让模块内的 hook 注册生效。

当前 hook 覆盖范围包括：

- `utils.common`
- `layers`，包括 attention、MoE、quantization、rotary embedding、linear 等
- `mem_cache`，包括 Kunlun KV pool 和 allocator
- `model_executor`
- `distributed`
- `constrained`
- `disaggregation`
- `speculative`
- `models`

### 4. Kernel Replacement 层

`sglang_kunlun.kernels.kernel_ops` 是 kernel 替换核心。它将每个替换点抽象为 `KernelSpec`：

```python
@dataclass(frozen=True)
class KernelSpec:
    module_path: str
    kernel_name: str
    impl: Callable
    metadata: Mapping[str, object] = field(default_factory=dict)
```

其中：

- `module_path` 表示上游模块路径。
- `kernel_name` 表示上游待替换符号。
- `impl` 表示 Kunlun 替代实现。
- `metadata` 用于描述安装策略，例如 `call_style = "direct"` 时直接替换，否则包装为 Triton 兼容 launcher。

注册入口分为两类：

- `register_triton_op()`：登记 Triton kernel 符号替换。
- `register_jit_op()`：登记 JIT/helper 符号替换。

Triton 上游常见调用形态是 `kernel[grid](*args, **kwargs)`，而 Kunlun 替代实现通常是普通 Python callable。`KernelLauncher` 会兼容该语法，并在替代实现需要 `grid` 参数时自动透传。

`install()` 会统一执行替换：

```text
_TRITON_OPS -> KernelLauncher 或 direct impl -> patch upstream symbol
_JIT_OPS    -> direct impl                   -> patch upstream symbol
```

替换时还会扫描已导入的 `sglang.*` 模块，将仍指向旧对象的 stale binding 一并替换，避免 `from xxx import old_symbol` 导致 patch 失效。

## MTP / speculative sampling 适配示例

上游 speculative / sampling 路径通常会调用 CUDA `sgl_kernel.sampling`：

```text
SGLang speculative / sampling
  -> import sgl_kernel.sampling
  -> top_k_renorm_prob / top_p_renorm_prob
  -> CUDA sgl_kernel implementation
```

Kunlun 环境下由 pre-shim 安装 `sgl_kernel` stub：

```text
python -m sglang_kunlun.launch
  -> 设置 SGLANG_PLATFORM=kunlun
  -> _kunlun_pre_shim()
  -> sgl_kernel_stub.install()
  -> sys.modules["sgl_kernel"] = Kunlun stub
  -> sys.modules["sgl_kernel.sampling"] = Kunlun stub
```

随后上游调用保持不变，但实际落到 Kunlun kernel 封装：

```text
sgl_kernel.sampling.top_k_renorm_prob(...)
  -> sglang_kunlun.kernels.sgl_kernel_kunlun.sampling.top_k_renorm_prob(...)
  -> kunlun_ops.top_k_renorm_probs(...)
```

这样 import 阶段和运行阶段分开处理，不需要直接修改上游 SGLang 调用点。

## 安装

在已准备好 SGLang、Kunlun runtime、`torch_xmlir`、`xspeedgate_ops`、`kunlun_ops` 等运行环境后，安装本插件：

```bash
pip install -e /home/zx/code/aicapx/sglang-kunlun
```

开发和测试环境可安装：

```bash
pip install -r /home/zx/code/aicapx/sglang-kunlun/requirements.txt
```

## 启动

推荐使用显式启动器，它会先设置 Kunlun 平台并执行 pre-shim：

```bash
unset XPU_DUMMY_EVENT
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_SET_CPU_AFFINITY=1
export XMLIR_FORCE_USE_XPU_GRAPH=1
export XPU_USE_FAST_SWIGLU=1
export XPU_USE_DEFAULT_CTX=1
export XMLIR_ENABLE_FAST_FC=1
export XMLIR_CUDNN_ENABLED=1
export CUDA_GRAPH_OPTIMIZE_STREAM=1
export SGLANG_PLATFORM=kunlun

SGLANG_ENABLE_SPEC_V2=1 sglang-kunlun-launch \
    --model-path /home/models/MiMo-V2-Flash-W8A8-INT8-Dynamic-official \
    --speculative-algorithm EAGLE \
    --quantization w8a8_int8 \
    --max-total-tokens 131072 \
    --disable-radix-cache \
    --decode-log-interval 1 \
    --host 0.0.0.0 \
    --port 8806 \
    --trust-remote-code \
    --tp-size 8 \
    --max-running-requests 64 \
    --disable-overlap-schedule \
    --attention-backend kunlun \
    --disable-cuda-graph \
    --mem-fraction-static 0.85
```

也可以在确认 pre-shim 已生效后直接使用 `sglang serve`：

```bash
export SGLANG_PLATFORM=kunlun

SGLANG_ENABLE_SPEC_V2=1 sglang serve \
    --model-path /home/models/MiMo-V2-Flash-W8A8-INT8-Dynamic-official \
    --speculative-algorithm EAGLE \
    --quantization w8a8_int8 \
    --max-total-tokens 131072 \
    --disable-radix-cache \
    --decode-log-interval 1 \
    --host 0.0.0.0 \
    --port 8806 \
    --trust-remote-code \
    --tp-size 8 \
    --max-running-requests 64 \
    --disable-overlap-schedule \
    --attention-backend kunlun \
    --disable-cuda-graph \
    --mem-fraction-static 0.85
```

## 新 kernel 替换接入

新增 Triton kernel 替换时，在 `sglang_kunlun/kernels/kernel_ops.py` 中声明 Kunlun 实现并注册：

```python
@register_triton_op("sglang.srt.mem_cache.common", "write_req_to_token_pool_triton")
def write_req_to_token_pool_triton(...):
    ...
```

这段代码在 import 时只会登记替换点，不会立即 patch。真正替换发生在 SGLang 加载插件并执行：

```text
load_plugins()
  -> entry point: sglang.srt.plugins / kunlun_hooks
  -> sglang_kunlun.hooks.registry.register_all()
  -> kernel_ops.install()
```

如果上游调用方式不是 `kernel[grid](...)`，可通过 metadata 指定 direct 替换：

```python
@register_triton_op(
    "some.upstream.module",
    "some_symbol",
    metadata={"call_style": "direct"},
)
def some_symbol(...):
    ...
```

新增 JIT/helper 替换时使用：

```python
@register_jit_op("some.upstream.module", "helper_name")
def helper_name(...):
    ...
```

## 新 hook 接入

新增函数、类或模块级行为替换时，建议放在对应子域目录下，例如：

- attention 相关：`sglang_kunlun/hooks/layers/attention/`
- KV cache 相关：`sglang_kunlun/hooks/mem_cache/`
- speculative 相关：`sglang_kunlun/hooks/speculative/`
- distributed 相关：`sglang_kunlun/hooks/distributed/`

新增模块后，需要在 `sglang_kunlun/hooks/registry.py` 的 `HOOK_MODULES` 中确保该模块会被导入，或者被已有包的 `__init__.py` 间接导入。hook 模块应保持 import-time 注册、runtime 生效的模式，避免在模块导入时执行重型初始化。

## 新模型接入

模型专项适配放在 `sglang_kunlun/models/` 下。SGLang 会根据模型 `config.json` 中的 `architectures` 字段匹配模型类，例如：

```json
{
  "architectures": ["DeepseekV4ForCausalLMDSpark"]
}
```

接入时需要保证：

- 外部包路径可被 import，例如 `sglang_kunlun.models`。
- 模型实现文件位于 `sglang_kunlun/models/` 下。
- 模块中暴露与 `architectures` 匹配的入口类。
- 必要的模型 hook 已通过 `registry.register_all()` 的导入链路注册。

## 测试

本仓库包含插件机制和兼容契约测试：

```bash
pytest /home/zx/code/aicapx/sglang-kunlun/test
```

重点测试包括：

- `test_pre_shim.py`：pre-shim 和 CUDA-like 兼容逻辑。
- `test_kernel_ops.py`：kernel replacement 注册、安装和 stale binding patch。
- `test_cuda_only_contract.py`：CUDA-only 依赖兼容契约。

## 与旧 monkey patch 方案的差异

相比 0.5.8 时代集中式 monkey patch 方案，本插件化方案的主要差异是：

- 加载边界更清晰：通过 SGLang OOT entry points 接入平台和通用插件。
- 职责拆分更明确：`platform`、`bootstrap/pre_shim`、`hooks/registry`、`kernels/kernel_ops`、`sgl_kernel_kunlun` 分层维护。
- patch 粒度更细：支持 kernel 符号级替换、函数级 hook、class 级替换和必要的模块级适配。
- import 时机更稳：`kernel_ops.install()` 会修补已导入到 `sglang.*` 命名空间中的旧符号引用。
- CUDA-only 依赖处理更完整：通过 `sgl_kernel` stub、Triton driver 兼容、`flashinfer/deep_gemm` hook 等方式承接上游 CUDA 假设。

## 排查建议

- 如果平台未被选中，检查 `SGLANG_PLATFORM=kunlun`、`torch_xmlir` 是否可导入，以及 `pip install -e` 后 entry points 是否生效。
- 如果 import 阶段出现 CUDA-only 依赖错误，优先确认是否通过 `sglang-kunlun-launch` 启动，或 `sitecustomize.py` 是否在当前 Python 环境中可见。
- 如果 attention backend 参数不接受 `kunlun`，确认 `sglang_kunlun.launch` 或 `KunlunSRTPlatform` 是否已执行 `_extend_attention_backend_choices()`。
- 如果 kernel 替换未生效，确认 `sglang_kunlun.hooks.registry.register_all()` 是否被 SGLang `load_plugins()` 调用，以及目标模块/符号名是否与当前 SGLang 版本一致。
