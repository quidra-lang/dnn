# dnn

`dnn` is Quidra's first-party deep-neural-network package. It provides layers,
activations, losses, and optimizers as ordinary Quidra source code on top of the
standard `tensor` and `neural` foundations.

## Install

dnn v0.1.0 supports Quidra `>=0.2.0 <0.3.0`.

With Quidra v0.2.0, install the exact released source:

```sh
git clone --depth 1 --branch v0.1.0 https://github.com/quidra-lang/dnn.git
cd dnn
quidra package install . --name dnn
```

Quidra versions that provide the release-aware short package CLI can install the
same immutable release directly:

```sh
quidra install dnn@0.1.0
```

Then import it normally. Factories validate their configuration and return
`error` for invalid dimensions or hyperparameters:

## API

Each layer factory returns a class value or `error`. Parameters and state remain
regular fields, so after validation a model is still an ordinary Quidra class:

| Factory | Returns | Methods |
| --- | --- | --- |
| `Linear(features_in, features_out, seed = 1)` | `LinearLayer | error` | `forward(value)` |
| `Conv2D(channels_in, channels_out, kernel, stride = 1, padding = 0, seed = 1)` | `Conv2DLayer | error` | `forward(value)` |
| `BatchNorm(features, momentum = 0.1, epsilon = 0.00001)` | `BatchNormLayer | error` | `forward(value)`, `infer(value)` |
| `Dropout(rate, seed = uint64(0))` | `DropoutLayer | error` | `forward(value)`, `infer(value)` |
| `SGD(rate = 0.01)` | `SGDOptimizer | error` | `step(&model, gradients)` |
| `Adam(rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 0.00000001)` | `AdamOptimizer | error` | `step(&model, gradients)` |

Validation is explicit: feature/channel/kernel sizes and strides must be
positive, convolution padding must be nonnegative, BatchNorm momentum and Adam
betas must be in `[0, 1)`, Dropout rate must be in `[0, 1)`, and optimizer rates
and epsilon values must be positive.

- Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`
- Losses: `mse`, `cross_entropy`, `binary_cross_entropy`, `binary_cross_entropy_with_logits`

`forward` builds a differentiable `neural<T>` value. `BatchNormLayer.forward`
updates the running statistics and `infer` is the read-only path over plain
tensors; `DropoutLayer.infer` returns its input unchanged. `LinearLayer.forward`
and `Conv2DLayer.forward` also accept a plain `tensor<float32>` when no gradient
is needed.

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
graph. `cross_entropy` accepts a one-hot `tensor<float32>` target with the same
shape as its logits. `gelu` uses the tanh approximation.

The layer factories use `float32`. Convolution expects NCHW inputs and OIHW
weights. Normalization treats axis 1 as the feature/channel axis.

`Linear` and `Conv2D` draw their weights uniformly from
`(-1/sqrt(fan_in), +1/sqrt(fan_in))`; biases start at zero. Initialization now
uses Quidra's standard explicit `random.Generator`, so DNN does not maintain a
second private RNG algorithm. The same `seed` reproduces the same weights and a
different seed produces a different layer. `Dropout` keeps its explicit neural
RNG state so it remains serializable with the model. There is no hidden global
random source.

`uniform_weights(count, bound, seed)` exposes the seeded initializer directly
and returns a rank-1 `tensor<float32>` of `count` values in `(-bound, +bound)`;
reshape it to build a layer with your own initialization, or construct
`LinearLayer` and `Conv2DLayer` from tensors you supply.

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

The released package dependency remains tied only to released Quidra versions.
During development, CI additionally builds the current Quidra `feature` branch
and checks the GPU placement/transfer contracts without changing
`requires.quidra` to an unreleased branch.

## Example

See [`examples/training.qui`](examples/training.qui).

## Development

See [`docs/development.md`](docs/development.md) for the canonical main/develop and release procedure.

## License

MIT
