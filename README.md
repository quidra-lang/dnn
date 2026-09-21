# Quidra DNN

Quidra DNN is Quidra's first-party deep-neural-network package, imported as
`dnn`. It owns DNN-specific
semantics and policy: layers, convolution/linear behavior, accelerator-library
selection, execution mode, activations, losses, and optimizers. The standard
`neural` namespace is intentionally the smaller autodiff/parameter foundation.

The current Core ABI still contains operand-level `neural.affine` and
`neural.convolve2d` hooks so ordinary Quidra source can attach these operations
to the built-in autograd graph without exposing backend details. Treat those
hooks as an implementation boundary, not as the long-term owner of DNN
semantics: new DNN operations belong here and should not expand `neural` unless
a genuinely generic autodiff primitive is required.

## Install

The package version and the Quidra range it requires are declared in
[`project.toml`](project.toml); [`quidra.package`](quidra.package) is generated
from it. Install a published release from a clone of its immutable tag:

```sh
git clone --depth 1 --branch <release-tag> https://github.com/quidra-lang/dnn.git
cd dnn
quidra install . --name dnn
```

Quidra versions that provide the release-aware short package CLI can install the
same immutable release directly:

```sh
quidra install dnn@<release-version>
```

Then import it normally:

```quidra
import dnn
```

Factories validate their configuration and return `error` for invalid dimensions
or hyperparameters.

## API

Every public top-level function has the following exact signature:

| Function | Signature |
| --- | --- |
| `fast` | `fast() -> void` |
| `deterministic` | `deterministic() -> void` |
| `all_reduce_sum` | `all_reduce_sum<T: floating>(tensor<T>[] &values) -> void` |
| `uniform_weights` | `uniform_weights(int count, float bound, int seed) -> tensor<float32> \| error` |
| `Linear` | `Linear(int features_in, int features_out, int seed = 1) -> LinearLayer \| error` |
| `Conv2D` | `Conv2D(int channels_in, int channels_out, int kernel, int stride = 1, int padding = 0, int seed = 1) -> Conv2DLayer \| error` |
| `BatchNorm` | `BatchNorm(int features, float momentum = 0.1, float epsilon = 0.00001) -> BatchNormLayer \| error` |
| `Dropout` | `Dropout(float rate, uint64 seed = uint64(0)) -> DropoutLayer \| error` |
| `relu` | `relu<X>(X value) -> X` |
| `tanh` | `tanh<X>(X value) -> X` |
| `sigmoid` | `sigmoid<X>(X value) -> X` |
| `softmax` | `softmax<X>(X value) -> X` |
| `gelu` | `gelu<X>(X value) -> X` |
| `mse` | `mse(neural<float32> prediction, tensor<float32> target) -> neural<float32>` |
| `binary_cross_entropy` | `binary_cross_entropy(neural<float32> prediction, tensor<float32> target) -> neural<float32>` |
| `binary_cross_entropy_with_logits` | `binary_cross_entropy_with_logits(neural<float32> logits, tensor<float32> target) -> neural<float32>` |
| `cross_entropy` | `cross_entropy(neural<float32> logits, tensor<float32> target) -> neural<float32>` |
| `SGD` | `SGD(float rate = 0.01) -> SGDOptimizer \| error` |
| `Adam` | `Adam(float rate = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 0.00000001) -> AdamOptimizer \| error` |

The public methods are:

| Class | Signature |
| --- | --- |
| `LinearLayer` | `forward<X>(X value) -> X` |
| `Conv2DLayer` | `forward<X>(X value) -> X` |
| `BatchNormLayer` | `forward(neural<float32> value) -> neural<float32>` |
| `BatchNormLayer` | `infer(tensor<float32> value) -> tensor<float32>` |
| `DropoutLayer` | `forward(neural<float32> value) -> neural<float32>` |
| `DropoutLayer` | `infer(tensor<float32> value) -> tensor<float32>` |
| `SGDOptimizer` | `step<M>(M &model, neural.Gradients gradients) -> void` |
| `AdamOptimizer` | `step<M>(M &model, neural.Gradients gradients) -> void` |

Each layer factory returns a class value or `error`. Parameters and state remain
regular fields, so after validation a model is still an ordinary Quidra class.

Validation is explicit: feature/channel/kernel sizes and strides must be
positive, convolution padding must be nonnegative, dimension products are
checked before multiplication, and floating configuration values must be finite.
BatchNorm momentum and Adam betas must be in `[0, 1)`, Dropout rate must be in
`[0, 1)`, and optimizer rates and epsilon values must be positive.
`uniform_weights` rejects negative counts and non-finite or negative bounds.

- Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`
- Losses: `mse`, `cross_entropy`, `binary_cross_entropy`, `binary_cross_entropy_with_logits`
- Multi-GPU: `all_reduce_sum(&values)` performs an in-place sum over same-shaped `float32`/`float` tensors on distinct NVIDIA GPUs

With a `neural<T>` input, `forward` extends the differentiable graph. `BatchNormLayer.forward`
updates the running statistics and `infer` is the read-only path over plain
tensors; `DropoutLayer.infer` returns its input unchanged. `LinearLayer.forward`
and `Conv2DLayer.forward` are generic over the input representation and also
accept a plain `tensor<float32>`, returning a plain tensor when no gradient is needed.

Training uses `neural.grad`; optimizer `step` methods mutate an explicitly
writable model. `AdamOptimizer` owns its iteration counter and moment state, so
saving the optimizer alongside the model preserves the state required to resume
training.

`softmax` subtracts the last-axis maximum and `cross_entropy` evaluates
log-softmax directly. `binary_cross_entropy` is for probability inputs and
keeps them strictly inside `(0, 1)`. For raw logits, prefer
`binary_cross_entropy_with_logits`, which uses the stable
`max(x, 0) - x*y + log(1 + exp(-abs(x)))` form and avoids exponentiating large
positive magnitudes. `sigmoid` is implemented through the bounded `tanh` path so
large negative inputs do not create an overflowing exponential in the backward
graph. `cross_entropy` is intended for class-distribution targets such as a one-hot
`tensor<float32>` with the same shape as the logits; ordinary Quidra tensor
broadcasting rules still apply to the target in the underlying arithmetic. `gelu`
uses the tanh approximation.

The layer factories use `float32`. Convolution expects NCHW inputs and OIHW
weights. Normalization treats axis 1 as the feature/channel axis.

`Linear` and `Conv2D` generate each weight from a `random.Generator.float()` sample `u` as
`float32((2*u - 1) / sqrt(fan_in))`; biases start at zero. Before the final
`float32` rounding, the sampled value lies in `[-1/sqrt(fan_in), +1/sqrt(fan_in))`. Initialization now
uses Quidra's standard explicit `random.Generator`, so DNN does not maintain a
second private RNG algorithm. The same `seed` reproduces the same weights and a
different seed produces a different layer. `Dropout` keeps its explicit neural
RNG state so it remains serializable with the model. There is no hidden global
random source.

`uniform_weights(count, bound, seed)` exposes the seeded initializer directly
and returns `tensor<float32> | error`: each valid output element is computed as
`float32((2*u - 1) * bound)` from `u = Generator.float()`. Before the final
`float32` rounding, the sampled value lies in `[-bound, +bound)`; `bound = 0`
produces zeros. Invalid count/bound configuration returns `error`. Reshape it to build a layer with your own initialization, or construct
`LinearLayer` and `Conv2DLayer` from tensors you supply.

## Execution mode

DNN is performance-first by default. Programs normally do not need to set a
mode: the default is `fast`, which permits the backend to select the fastest
supported algorithms, including nondeterministic algorithms when they are
faster.

For reproducibility-sensitive runs, switch the process-wide DNN execution mode
once near program startup:

```quidra
dnn.deterministic()
```

To switch back explicitly:

```quidra
dnn.fast()
```

The execution policy is deliberately exposed as these two direct calls rather
than accepting an arbitrary function value. `deterministic()` restricts
accelerated backends to deterministic algorithms; `fast()` permits the backend
to choose the fastest valid algorithm. Explicit Quidra RNG state, such as a Dropout seed, remains
separate from algorithm determinism and is never replaced by a hidden GPU RNG.

## Device placement

DNN follows Quidra tensor placement exactly. It never inserts CPU↔GPU or
GPU↔GPU transfers on behalf of a layer, loss, optimizer, or model. CPU remains
the default, and callers opt into a GPU explicitly with Quidra's ordinary
tensor surface:

```quidra
tensor<float32> input = tensor.zeros<float32>([32, 128], gpu = 0)
tensor<float32> copied = tensor.ones<float32>([32, 128]).gpu(0)
tensor<float32> host = copied.cpu()
```

A DNN operation must consume parameters/state on a compatible device and return
its result on that same device. If the active DNN backend does not implement the
requested GPU operation yet, execution fails explicitly instead of copying the
tensor to CPU. The package surface stays vendor-independent; NVIDIA acceleration
belongs behind the DNN backend boundary (cuDNN/cuBLAS and, where appropriate,
NCCL), Apple acceleration behind the Metal backend, and AMD acceleration behind
the ROCm/HIP backend.

DNN owns the public accelerator policy and the release contract for
cuDNN/cuBLAS/NCCL selection. In the current source-package architecture, the
native loader and backend ABI are still compiled into the Quidra runtime; this
is an implementation boundary rather than a second public DNN API. A packaged
DNN release is expected to pin the NVIDIA component versions, artifacts, and
checksums it supports. The runtime never searches a system-installed
cuDNN/cuBLAS/NCCL by default. An installed DNN package is automatically probed
for a managed bundle under `~/.quidra/packages/dnn/nvidia/lib` on POSIX or
`%USERPROFILE%\.quidra\packages\dnn\nvidia\lib` on Windows.
`QUIDRA_DNN_NVIDIA_LIBRARY_PATH` is a strict explicit override: when it is set,
failure to load from that directory does not silently fall through to another
bundle. Development environments may explicitly opt into system libraries with
`QUIDRA_DNN_ALLOW_SYSTEM_NVIDIA_LIBS=1`. The bundle layout and release
verification contract are documented in `nvidia/README.md`. Quidra core owns
device/driver execution and the current native bridge, while DNN owns
neural-library policy and the source-visible surface.

On NVIDIA GPUs, the current development backend dispatches Linear/affine forward
and backward through cuBLAS when available, and Conv2D forward/backward through
cuDNN. `fast` selects from cuDNN's non-executing algorithm heuristics and caches
the selected algorithm per device/shape; `deterministic` excludes nondeterministic
cuDNN algorithms. If an accelerator operation is unavailable, the same operation
falls back to Quidra's native GPU implementation, never to CPU. NCCL backs
`all_reduce_sum(&values)` for explicit same-process multi-GPU tensor reduction.

The release workflow derives the required Core baseline tag from the lower bound
of `requires.quidra` and checks that the tag exists before building or tagging
DNN.
During development, CI additionally builds the current Quidra `develop` branch
and checks GPU placement, inference, autograd, and optimizer contracts without
changing `requires.quidra` to an unreleased branch. On a machine with a real
GPU, `tests/real_gpu_integration.sh /path/to/quidra` runs CPU↔GPU numerical
equivalence checks; set `QUIDRA_REQUIRE_REAL_GPU=1` in a hardware runner to
make absence of a real GPU a test failure.

## Example

See [`examples/training.qui`](examples/training.qui).

## Development

See [`docs/development.md`](docs/development.md) for the canonical main/develop and release procedure.

## License

MIT
