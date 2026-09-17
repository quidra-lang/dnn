# dnn

`dnn` is Quidra's first-party deep-neural-network package. It provides layers,
activations, losses, and optimizers as ordinary Quidra source code on top of the
standard `tensor` and `neural` foundations.

## Install

Clone this repository and install it with the Quidra package command:

```sh
quidra package install . --name dnn
```

Then import it normally:

```quidra
import dnn

dnn.LinearLayer layer = dnn.Linear(features_in = 2, features_out = 1)
tensor<float32> samples = tensor.ones<float32>([1, 2])
neural<float32> prediction = layer.forward(neural.track(samples))
```

For development, place the repository in a directory listed by
`QUIDRA_PACKAGE_PATH` instead of installing it.

## API

Each layer factory returns a class value whose parameters and state are regular
fields, so a model is an ordinary Quidra class:

| Factory | Returns | Methods |
| --- | --- | --- |
| `Linear(features_in, features_out, seed = 1)` | `LinearLayer` | `forward(value)` |
| `Conv2D(channels_in, channels_out, kernel, stride = 1, padding = 0, seed = 1)` | `Conv2DLayer` | `forward(value)` |
| `BatchNorm(features, momentum = 0.1, epsilon = 0.00001)` | `BatchNormLayer` | `forward(value)`, `infer(value)` |
| `Dropout(rate, seed = uint64(0))` | `DropoutLayer` | `forward(value)`, `infer(value)` |
| `SGD(rate = 0.01)` | `SGDOptimizer` | `step(&model, gradients)` |
| `Adam(rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 0.00000001)` | `AdamOptimizer` | `step(&model, gradients)` |

- Activations: `relu`, `sigmoid`, `tanh`, `softmax`, `gelu`
- Losses: `mse`, `cross_entropy`, `binary_cross_entropy`

`forward` builds a differentiable `neural<T>` value. `BatchNormLayer.forward`
updates the running statistics and `infer` is the read-only path over plain
tensors; `DropoutLayer.infer` returns its input unchanged. `LinearLayer.forward`
and `Conv2DLayer.forward` also accept a plain `tensor<float32>` when no gradient
is needed.

Training uses `neural.grad`; optimizer `step` methods mutate an explicitly
writable model. `AdamOptimizer` owns its iteration counter and moment state, so
saving the optimizer alongside the model preserves a resumable training run.

`softmax` subtracts the last-axis maximum, `cross_entropy` evaluates
log-softmax directly, and `binary_cross_entropy` keeps its probability strictly
inside `(0, 1)`, so saturated predictions and extreme logits stay finite.
`cross_entropy` accepts a one-hot `tensor<float32>` target with the same shape
as its logits. `tanh` saturates instead of overflowing, and `gelu` uses the
tanh approximation.

The layer factories use `float32`. Convolution expects NCHW inputs and OIHW
weights. Normalization treats axis 1 as the feature/channel axis.

`Linear` and `Conv2D` draw their weights uniformly from
`(-1/sqrt(fan_in), +1/sqrt(fan_in))` so that the units of a layer start out
different and can learn independently; biases start at zero. The draw comes
from an integer generator seeded by `seed`, so the same `seed` always produces
the same weights and a different `seed` produces a different layer. `Dropout`
takes its own `seed` the same way. Runs are therefore reproducible by default
rather than randomized by a hidden global source.

`uniform_weights(count, bound, seed)` exposes that generator directly and
returns a rank-1 `tensor<float32>` of `count` values in `(-bound, +bound)`;
reshape it to build a layer with your own initialization, or construct
`LinearLayer` and `Conv2DLayer` from tensors you supply.

## Example

See [`examples/training.qui`](examples/training.qui).

## License

MIT
