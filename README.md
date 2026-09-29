# Quidra DNN

Quidra DNN is Quidra's first-party deep-neural-network package, imported as
`dnn`. It owns DNN-specific
semantics and policy: layers, convolution/fully-connected behavior, accelerator-library
selection, execution mode, activations, losses, and optimizers. Autograd lives directly on floating-point `tensor<T>` values. DNN owns `Parameter<T>`, `State<T>`, SGD/Adam, model/training persistence policy, and distributed-training policy; Core remains limited to tensor/autograd and generic tensor primitives.

DNN layers are implemented from generic Core tensor primitives rather than
DNN-specific Core operations. `FC` uses `tensor.matmul()`; `Conv2D`
uses mathematical `tensor.convolve()` plus tensor gather/scatter to implement
DNN cross-correlation and channel mixing. DNN owns learnable `Parameter` storage and non-trainable persistent `State` storage; generic tensor/autograd semantics remain in Core.

## Install

The package version and the Quidra range it requires are declared in
[`project.toml`](project.toml); [`quidra.package`](quidra.package) is generated
from it. Install a published release from a clone of its immutable tag:

```sh
git clone --depth 1 --branch <release-tag> https://github.com/quidra-lang/dnn.git
cd dnn
quidra install .
```

Quidra versions that provide the release-aware short package CLI can install the
same immutable release directly:

```sh
quidra install quidra-dnn@<release-version>
```

The package-manager identity is `quidra-dnn`; the Quidra source import
identifier remains `dnn`. This distinction keeps installation names globally
recognizable without making source imports longer.

Then import it normally:

```quidra
import dnn
```

Factories validate their configuration and return `error` for invalid dimensions
or hyperparameters.

Model persistence is DNN-owned: `dnn.save(model, path = "model.dnn")` serializes only `Parameter<float32/float>` and `State<float32/float>` fields discovered through `reflect.collect`, and `dnn.load(&model, path = "model.dnn")` restores those existing storages. Ordinary tensor/scalar fields are intentionally not part of model state. Loading validates all serialized tensors before replacing any existing Parameter or State storage. `Adam.save(path)` persists iteration and first/second moments; `Adam.load(&model, path)` validates the target parameter binding and moment shapes before restoring optimizer state, so training can resume without moving optimizer policy into Core.

## API

The primary source-visible top-level API has the following exact signatures:

| Function | Signature |
| --- | --- |
| `mode.fast` | `mode.fast() -> void` |
| `mode.deterministic` | `mode.deterministic() -> void` |
| `save` | `save<M>(M model, string path) -> void \| error` |
| `load` | `load<M>(M &model, string path) -> void \| error` |
| `all_reduce_sum` | `all_reduce_sum<T: floating>(tensor<T>[] &values, int[] devices) -> void \| error` |
| `uniform_weights` | `uniform_weights(int count, float bound, int seed) -> tensor<float32> \| error` |
| `normal_weights` | `normal_weights(int count, float standard_deviation, int seed) -> tensor<float32> \| error` |
| `normal_fc` | `normal_fc(int features_in, int features_out, float standard_deviation, float bias_value = 0.0, int seed = 1) -> FC \| error` |
| `normal_conv2d` | `normal_conv2d(int channels_in, int channels_out, int kernel, float standard_deviation, float bias_value = 0.0, int stride = 1, int padding = 0, int seed = 1, int groups = 1) -> Conv2D \| error` |
| `max_pool2d` | `max_pool2d(tensor<float32> value, int kernel, int stride, int padding = 0) -> tensor<float32>` |
| `global_average_pool2d` | `global_average_pool2d(tensor<float32> value) -> tensor<float32>` |
| `local_response_normalization` | `local_response_normalization(tensor<float32> value, int size = 5, float alpha = 0.0001, float beta = 0.75, float k = 2.0) -> tensor<float32>` |
| `relu` | `relu(tensor<float32> value) -> tensor<float32>` |
| `tanh` | `tanh(tensor<float32> value) -> tensor<float32>` |
| `sigmoid` | `sigmoid(tensor<float32> value) -> tensor<float32>` |
| `softmax` | `softmax(tensor<float32> value) -> tensor<float32>` |
| `gelu` | `gelu(tensor<float32> value) -> tensor<float32>` |
| `mse` | `mse(tensor<float32> prediction, tensor<float32> target) -> tensor<float32>` |
| `binary_cross_entropy` | `binary_cross_entropy(tensor<float32> prediction, tensor<float32> target) -> tensor<float32>` |
| `binary_cross_entropy_with_logits` | `binary_cross_entropy_with_logits(tensor<float32> logits, tensor<float32> target) -> tensor<float32>` |
| `cross_entropy` | `cross_entropy(tensor<float32> logits, tensor<float32> target) -> tensor<float32>` |

Layers and optimizers are classes. Configuration-validating layer/optimizer constructors can fail: a bare failing call such as `dnn.FC(2, 0)` fails fast because a zero output width is invalid; inside a fallible function, `try dnn.FC(2, 0)` propagates that error; and `dnn.FC | error layer = dnn.FC(2, 0)` retains it for a `match`. `Parameter<T>` and `State<T>` have ordinary infallible constructors because they only wrap existing tensors. Arguments may be positional or named.

| Class | Constructor |
| --- | --- |
| `Parameter<T>` | `Parameter(tensor<T> value)` |
| `State<T>` | `State(tensor<T> value)` |
| `FC` | `FC(int features_in, int features_out, int seed = 1)` |
| `Conv2D` | `Conv2D(int channels_in, int channels_out, int kernel, int stride = 1, int padding = 0, int seed = 1, int groups = 1)` |
| `BatchNorm` | `BatchNorm(int features, float momentum = 0.1, float epsilon = 0.00001)` |
| `Dropout` | `Dropout(float rate, int seed = 0, bool inverted = true)` |
| `SGD` | `SGD(float rate = 0.01)` |
| `Adam` | `Adam(float rate = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 0.00000001)` |

The public methods are:

| Class | Signature |
| --- | --- |
| `FC` | `forward(tensor<float32> value) -> tensor<float32>` |
| `Conv2D` | `forward(tensor<float32> value) -> tensor<float32>` |
| `BatchNorm` | `forward(tensor<float32> value) -> tensor<float32>` |
| `BatchNorm` | `infer(tensor<float32> value) -> tensor<float32>` |
| `Dropout` | `forward(tensor<float32> value) -> tensor<float32>` |
| `Dropout` | `infer(tensor<float32> value) -> tensor<float32>` |
| `SGD` | `zero_grad<M>(M &model) -> void`; `step<M>(M &model) -> void` |
| `Adam` | `zero_grad<M>(M &model) -> void`; `step<M>(M &model) -> void`; `save(string path) -> void \| error`; `load<M>(M &model, string path) -> void \| error` |

## Reference ImageNet architectures

Complete paper-reference architectures are grouped under `dnn.model`; reusable layers, losses, optimizers, and persistence remain directly under `dnn`. Reference-only building blocks used to reproduce those papers are grouped under `dnn.model` as well, rather than leaking into the root layer API.

DNN includes source-level reference implementations of the original dnn.model.AlexNet,
VGG, and ResNet ImageNet architectures. These classes preserve the papers'
learned-layer topology, channel widths, stage repetition counts, pooling
geometry, and classifier widths rather than substituting later library variants.
`forward` returns logits for use with `cross_entropy`; `probabilities`
applies the papers' terminal 1000-way softmax to `infer` output.

- `dnn.model.AlexNet` consumes 224x224 RGB input and uses 96@11x11/4,
  256@5x5, 384@3x3, 384@3x3, and 256@3x3 convolutions. Convolutions
  2, 4, and 5 use two groups to reproduce the original two-GPU connectivity;
  LRN follows conv1 and conv2; 3x3/2 overlapping max-pooling follows both LRN
  stages and conv5; the classifier is 9216 -> 4096 -> 4096 -> 1000. The first
  convolution uses a two-pixel border so the paper's explicit 224x224 input
  reaches its published 55 -> 27 -> 13 -> 6 spatial sequence. Dropout uses the
  original non-inverted rule: 0.5 dropping during training and 0.5 scaling at
  inference.
- The VGG paper configurations are exposed directly as `dnn.model.VGGA`, `dnn.model.VGGALRN`,
  `dnn.model.VGGB`, `dnn.model.VGGC`, `dnn.model.VGGD`, and `dnn.model.VGGE`. The conventional names
  `dnn.model.VGG11`, `dnn.model.VGG13`, `dnn.model.VGG16`, and `dnn.model.VGG19` map to A, B, D, and E.
  All consume 224x224 RGB input, use five 2x2/2 max-pooling stages, and end in
  25088 -> 4096 -> 4096 -> 1000. Configuration C retains the paper's 1x1
  layers, and A-LRN retains its LRN variant instead of silently collapsing the
  table to only VGG-16/19.
- `dnn.model.ResNet18`, `dnn.model.ResNet34`, `dnn.model.ResNet50`, `dnn.model.ResNet101`, and `dnn.model.ResNet152`
  use the paper's 7x7/2, 64-channel stem, 3x3/2 max-pool, four residual stages,
  global average pool, and 1000-way FC. The basic-block models use stage counts
  [2,2,2,2] and [3,4,6,3]. To match the paper's parameter-free comparison
  networks, ResNet-18/34 use Option A at dimension changes: stride-2 identity
  sampling plus zero-padded channels. The bottleneck models use
  [3,4,6,3], [3,4,23,3], and [3,8,36,3] with 1x1-3x3-1x1 bottlenecks and
  Option B projection shortcuts only when dimensions increase; the stage-changing
  stride is on the first 1x1 convolution as in the original ResNet-v1 paper.
  Repeated residual blocks live in runtime arrays inside one generalized stage class per block family, so depths such as 6, 23, and 36 are constructor counts rather than separate `Stage6`, `Stage23`, or `Stage36` types.

The reference constructors use the initialization stated by the corresponding
original work instead of DNN's generic fan-in uniform layer default. dnn.model.AlexNet
samples every learned weight from a zero-mean Gaussian with standard deviation
0.01; conv2/conv4/conv5 and the two hidden FC biases start at 1, while the
remaining biases start at 0. VGG's paper literally specifies a zero-mean
Gaussian with variance 10^-2 and zero biases, so its random path uses standard
deviation sqrt(10^-2) = 0.1 rather than silently reinterpreting "variance" as
"standard deviation". ResNet follows the paper's reference to the He/MSRA
initializer: zero-mean Gaussian weights with standard deviation sqrt(2/fan_in)
and zero convolution/FC biases. Its BatchNorm scale/bias already start at 1/0.

VGG's deeper B/C/D/E training initialization has one extra paper dependency: it
starts from a trained configuration A for A's four shape-compatible convolution
layers and all three FC layers, while the additional layers retain random
initialization. After training/loading a `dnn.model.VGGA`, call
`deeper.initialize_from(vgg_a)`; the `dnn.model.VGG13`, `dnn.model.VGG16`, and `dnn.model.VGG19`
aliases expose the same operation with a `dnn.model.VGG11` source. A constructor cannot
invent those learned A values from a random seed, so this transfer remains
explicit instead of pretending a random draw is the paper's pretrained state.
Dataset preprocessing, training schedules, optimizer hyperparameters, and
published checkpoints remain separate from model construction.

`Parameter<T>` and `State<T>` are ordinary DNN classes backed by explicit shared `ref.Cell` value storage. Both cut any incoming autograd graph when they take ownership of a tensor. A Parameter additionally owns a private Core `autograd.Target` for its gradient destination; State owns no gradient destination. `Parameter.track()` starts a graph leaf bound to that target only when a tracked forward path needs the Parameter. Parameters are trainable and optimizer-visible; State is persistent but never optimizer-updated. BatchNorm uses `State<float32>` for running mean and variance, while configuration scalars such as momentum and epsilon remain ordinary non-persistent fields.

Validation is explicit: feature/channel/kernel sizes and strides must be
positive, convolution padding must be nonnegative, dimension products are
checked before multiplication, and floating configuration values must be finite.
BatchNorm momentum and Adam betas must be in `[0, 1)`, Dropout rate must be in
`[0, 1)`, and optimizer rates and epsilon values must be positive.
`uniform_weights` rejects negative counts and non-finite or negative bounds. `normal_weights` likewise rejects negative counts and negative/non-finite standard deviations; the normal layer factories also reject non-finite biases.

- Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`
- Losses: `mse`, `cross_entropy`, `binary_cross_entropy`, `binary_cross_entropy_with_logits`

A floating-point tensor carries an autograd graph only after `.track()` (or
when it is derived from a tracked tensor). Passing an ordinary untracked tensor
through a layer keeps Parameters as raw constants and does not create a graph;
a tracked input causes only the Parameters that intersect that tracked path to
enter the graph. This autograd participation is independent of layer behavior:
`BatchNorm.forward` updates running statistics while `BatchNorm.infer` does not,
and `Dropout.forward` applies training behavior while `Dropout.infer` returns
its input unchanged. A tracked input may therefore still be used with an
`infer` method when gradients are desired without training-time state behavior.

Training computes the tracked forward path and loss first, then places the
three gradient/parameter mutations together:

```quidra
tensor<float32> prediction = model.forward(input.track())
tensor<float32> loss = dnn.mse(prediction, target)

optimizer.zero_grad(&model)
loss.backward(&model)
optimizer.step(&model)
```

This ordering makes the accumulation boundary visible exactly where gradients
are about to be written. Move `zero_grad(&model)` outside a multi-batch loop
when accumulation is intentional; repeated `backward` calls accumulate until
`zero_grad(&model)` or `clear_grad()` clears the destination.

`backward` writes only to Parameters reachable from the explicitly named model
targets and actually present in the loss graph; `loss.backward(&encoder)` does
not write decoder or input gradients, while
`loss.backward(&encoder, &decoder, &input)` names all three destinations
explicitly. Parameter gradients live in the Parameter's DNN-owned
`autograd.Target` and are exposed through `gradient()`. `State<T>` is never a
backward, zero_grad, or optimizer-step destination. `Adam` owns its iteration
counter and moment state, so saving the optimizer alongside the model preserves
the state required to resume training.

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

The layer constructors use `float32`. Convolution expects NCHW inputs and OIHW
weights. Normalization treats axis 1 as the feature/channel axis.

`FC` and `Conv2D` generate each weight from a `random.Generator.float()` sample `u` as
`float32((2*u - 1) / sqrt(fan_in))`; biases start at zero. Before the final
`float32` rounding, the sampled value lies in `[-1/sqrt(fan_in), +1/sqrt(fan_in))`. Initialization now
uses Quidra's standard explicit `random.Generator`, so DNN does not maintain a
second private RNG algorithm. The same `seed` reproduces the same weights and a
different seed produces a different layer. `Dropout` also uses an explicit
`random.Generator`, so it no longer depends on a DNN-specific random-mask Core
primitive. There is no hidden global random source.

`uniform_weights(count, bound, seed)` exposes the seeded initializer directly
and returns `tensor<float32> \| error`: each valid output element is computed as
`float32((2*u - 1) * bound)` from `u = Generator.float()`. Before the final
`float32` rounding, the sampled value lies in `[-bound, +bound)`; `bound = 0`
produces zeros. Invalid count/bound configuration returns `error`. Reshape it to build a layer with your own initialization, or declare a
`FC` or `Conv2D` without a constructor call and assign the tensors you supply.

`normal_weights(count, standard_deviation, seed)` is the zero-mean Gaussian
counterpart. It uses Box-Muller over Quidra's explicit `random.Generator`, is
seed-reproducible, and rejects negative/non-finite standard deviations.
`normal_fc` and `normal_conv2d` construct layers directly from that Gaussian
without first generating and discarding the generic uniform weights; they also
make the requested constant bias explicit. The paper-reference models use these
factories so large VGG classifiers do not pay for a throwaway initialization.

## Execution mode

DNN is performance-first by default. Programs normally do not need to set a
mode: the default is `fast`, which permits the backend to select the fastest
supported algorithms, including nondeterministic algorithms when they are
faster.

For reproducibility-sensitive runs, switch the process-wide DNN execution mode
once near program startup:

```quidra
dnn.mode.deterministic()
```

To switch back explicitly:

```quidra
dnn.mode.fast()
```

The execution policy is deliberately exposed as these two direct calls in the
`dnn.mode` namespace rather than accepting an arbitrary function value.
`mode.deterministic()` restricts accelerated backends to deterministic
algorithms; `mode.fast()` permits the backend to choose the fastest valid
algorithm. Explicit Quidra RNG state, such as a Dropout seed, remains
separate from algorithm determinism and is never replaced by a hidden GPU RNG.

## Device placement

DNN follows Quidra tensor placement exactly. Layers, losses, optimizers, and
models never insert CPU↔GPU or GPU↔GPU transfers implicitly. CPU remains the
default, and callers opt into a GPU explicitly with Quidra's ordinary tensor
surface:

```quidra
tensor<float32> input = tensor.zeros([32, 128], gpu = 0)
tensor<float32> copied = tensor.ones([32, 128]).gpu(0)
tensor<float32> host = copied.cpu()
```

A DNN operation must consume parameters/state on a compatible device and return
its result on that same device. If the active DNN backend does not implement the
requested GPU operation yet, execution fails explicitly instead of copying the
tensor to CPU. The package surface stays vendor-independent; NVIDIA acceleration
belongs behind the DNN backend boundary (cuDNN/cuBLAS and, where appropriate,
NCCL), Apple acceleration behind the Metal backend, and AMD acceleration behind
the ROCm/HIP backend.

Distributed reduction is an explicit exception because the transfer itself is
the requested operation. DNN exposes `all_reduce_sum(&values, devices)`, where
each destination is `-1` for CPU or a nonnegative GPU index. Inputs must be
untracked, have the same dtype and shape, and the result sum replaces every
entry on its requested destination. The baseline implementation is host-staged
through ordinary `.cpu()` / `.gpu(n)` transfers, so Core needs no
DNN-specific collective API. A vendor backend may accelerate this transport
(for example with NCCL) without changing the DNN-visible semantics.

```quidra
tensor<float32>[] gradients = [first_gradient, second_gradient]
dnn.all_reduce_sum(&gradients, [0, 1])
```

DNN owns the public accelerator policy and the release contract for
cuDNN/cuBLAS/distributed accelerator selection. In the current source-package architecture, the
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
DNN-library policy and the source-visible surface.

On GPUs, `FC` and `Conv2D` are expressed through Core tensor primitives
rather than opaque layer kernels. `FC` lowers to batched `tensor.matmul`;
`Conv2D` composes differentiable `tensor.convolve()` calls with generic
`gather`/`scatter`, so device placement and autograd stay in the shared tensor
runtime. CUDA matrix multiplication uses cuBLAS when available; the generic GPU
fallback never silently transfers the operation to CPU.

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

[`examples/training.qui`](examples/training.qui) is the minimal end-to-end
training example. User-facing examples normally use Quidra's fail-fast success
context for fallible constructors so the learning flow stays visible; explicit
`error` handling is shown only when handling failure is the point of the
example.

## Development

See [`docs/development.md`](docs/development.md) for the canonical main/develop and release procedure.

## License

MIT
