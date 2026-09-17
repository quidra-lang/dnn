#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/use-dnn.qui" <<'QUI'
import dnn

class Model
    dnn.LinearLayer dense

int | error run()
    dnn.LinearLayer dense = try dnn.Linear(features_in = 2, features_out = 1)
    Model model = Model(dense = dense)
    tensor<float32> samples = tensor.ones<float32>([1, 2])
    tensor<float32> targets = tensor.zeros<float32>([1, 1])
    neural<float32> prediction = dnn.relu(
        model.dense.forward(neural.track(samples))
    )
    neural<float32> loss = dnn.mse(prediction, targets)
    neural.Gradients gradients = neural.grad(loss)
    dnn.SGDOptimizer optimizer = try dnn.SGD(rate = 0.1)
    optimizer.step(&model, gradients)

    print(prediction.untrack().shape()[0])
    print(prediction.untrack().shape()[1])

    float32 first = loss.untrack().item()
    float32 latest = first
    for iteration in range(20)
        neural<float32> step_prediction = model.dense.forward(
            neural.track(samples)
        )
        neural<float32> step_loss = dnn.mse(step_prediction, targets)
        neural.Gradients step_gradients = neural.grad(step_loss)
        optimizer.step(&model, step_gradients)
        latest = step_loss.untrack().item()
    print(latest < first)
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/use-dnn.qui")"
expected="$(printf '1\n1\ntrue')"
if [[ "$output" != "$expected" ]]; then
    echo "unexpected dnn output: $output" >&2
    exit 1
fi

cat > "$TMP/operations.qui" <<'QUI'
import dnn

int | error run()
    dnn.Conv2DLayer convolution = try dnn.Conv2D(
        channels_in = 1,
        channels_out = 2,
        kernel = 3,
        padding = 1
    )
    tensor<float32> pixels = tensor.ones<float32>([1, 1, 4, 5])
    tensor<float32> filtered = convolution.forward(pixels)
    print(filtered.shape()[0])
    print(filtered.shape()[1])
    print(filtered.shape()[2])
    print(filtered.shape()[3])

    tensor<float32> values = tensor.zeros<float32>([1, 2])
    values[0, 0] = float32(-1)
    values[0, 1] = float32(1)
    tensor<float32> activated = dnn.relu(values)
    tensor<float32> probabilities = dnn.softmax(values)
    print(activated[0, 0].item())
    print(activated[0, 1].item())
    print(math.abs(float(probabilities[0, 0].item() + probabilities[0, 1].item()) - 1.0) < 0.000001)
    print(dnn.sigmoid(values)[0, 0].item() > float32(0))
    print(dnn.tanh(values)[0, 0].item() < float32(0))
    print(dnn.gelu(values)[0, 1].item() > float32(0))

    tensor<float32> target = tensor.zeros<float32>([1, 2])
    target[0, 1] = float32(1)
    neural<float32> logits = neural.track(values)
    print(dnn.cross_entropy(logits, target).untrack().item() > float32(0))
    print(dnn.binary_cross_entropy(dnn.sigmoid(logits), target).untrack().item() > float32(0))
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

operations_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/operations.qui")"
operations_expected="$(printf '1\n2\n4\n5\n0.0\n1.0\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$operations_output" != "$operations_expected" ]]; then
    echo "unexpected dnn operations output: $operations_output" >&2
    exit 1
fi

cat > "$TMP/stateful-layers.qui" <<'QUI'
import dnn

int | error run()
    dnn.BatchNormLayer normalization = try dnn.BatchNorm(features = 2)
    tensor<float32> values = tensor.ones<float32>([2, 2])
    neural<float32> trained = normalization.forward(neural.track(values))
    tensor<float32> inferred = normalization.infer(values)
    print(trained.untrack().shape()[1])
    print(inferred.shape()[1])
    print(normalization.running_mean.value[0].item() > float32(0))

    dnn.DropoutLayer masking = try dnn.Dropout(rate = 0.5, seed = uint64(17))
    neural<float32> masked = masking.forward(neural.track(values))
    print(masked.untrack().shape()[0])
    print(masking.rng.value != uint64(17))
    print(masking.infer(values)[0, 0].item())
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

stateful_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/stateful-layers.qui")"
stateful_expected="$(printf '2\n2\ntrue\n2\ntrue\n1.0')"
if [[ "$stateful_output" != "$stateful_expected" ]]; then
    echo "unexpected dnn stateful output: $stateful_output" >&2
    exit 1
fi

cat > "$TMP/adam.qui" <<'QUI'
import dnn

class Model
    dnn.LinearLayer dense

int | error run()
    dnn.LinearLayer dense = try dnn.Linear(features_in = 1, features_out = 1)
    Model model = Model(dense = dense)
    dnn.AdamOptimizer optimizer = try dnn.Adam(rate = 0.1)
    tensor<float32> samples = tensor.ones<float32>([1, 1])
    tensor<float32> targets = tensor.ones<float32>([1, 1])
    float32 before = model.dense.weight.raw()[0, 0].item()
    neural<float32> prediction = model.dense.forward(neural.track(samples))
    neural<float32> loss = dnn.mse(prediction, targets)
    neural.Gradients gradients = neural.grad(loss)
    optimizer.step(&model, gradients)
    print(optimizer.iteration.value)
    print(model.dense.weight.raw()[0, 0].item() > before)
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

adam_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/adam.qui")"
if [[ "$adam_output" != "$(printf '1\ntrue')" ]]; then
    echo "unexpected dnn Adam output: $adam_output" >&2
    exit 1
fi

cat > "$TMP/numeric-stability.qui" <<'QUI'
import dnn

class Model
    dnn.LinearLayer dense

int | error run()
    tensor<float32> extreme = tensor.zeros<float32>([1, 2])
    extreme[0, 0] = float32(50.0)
    extreme[0, 1] = float32(-50.0)
    print(dnn.tanh(extreme)[0, 0].item())
    print(dnn.tanh(extreme)[0, 1].item())
    print(dnn.gelu(extreme)[0, 0].item())

    tensor<float32> saturated = tensor.zeros<float32>([1, 2])
    saturated[0, 0] = float32(1.0)
    saturated[0, 1] = float32(0.0)
    tensor<float32> labels = tensor.zeros<float32>([1, 2])
    labels[0, 0] = float32(1.0)
    neural<float32> saturated_loss = dnn.binary_cross_entropy(
        neural.track(saturated), labels
    )
    print(saturated_loss.untrack().item() < float32(0.001))

    tensor<float32> logits = tensor.zeros<float32>([1, 2])
    logits[0, 1] = float32(200.0)
    tensor<float32> target = tensor.zeros<float32>([1, 2])
    target[0, 0] = float32(1.0)
    print(dnn.cross_entropy(neural.track(logits), target).untrack().item())

    dnn.LinearLayer dense = try dnn.Linear(features_in = 2, features_out = 2)
    Model model = Model(dense = dense)
    tensor<float32> samples = tensor.ones<float32>([1, 2])
    tensor<float32> classes = tensor.zeros<float32>([1, 2])
    classes[0, 1] = float32(1)
    neural<float32> scores = model.dense.forward(neural.track(samples))
    neural<float32> loss = dnn.cross_entropy(scores, classes)
    neural.Gradients gradients = neural.grad(loss)
    float32 before = model.dense.bias.raw()[1].item()
    dnn.SGDOptimizer optimizer = try dnn.SGD(rate = 0.5)
    optimizer.step(&model, gradients)
    print(model.dense.bias.raw()[1].item() > before)
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

stability_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/numeric-stability.qui")"
stability_expected="$(printf '1.0\n-1.0\n50.0\ntrue\n200.0\ntrue')"
if [[ "$stability_output" != "$stability_expected" ]]; then
    echo "unexpected dnn numeric stability output: $stability_output" >&2
    exit 1
fi

cat > "$TMP/initialization.qui" <<'QUI'
import dnn

int | error run()
    dnn.LinearLayer layer = try dnn.Linear(features_in = 3, features_out = 4)
    print(layer.weight.raw().shape()[0])
    print(layer.weight.raw().shape()[1])

    int identical = 0
    for unit in range(1, 4)
        int matching = 0
        for feature in range(3)
            if layer.weight.raw()[unit, feature].item() == layer.weight.raw()[0, feature].item()
                matching += 1
        if matching == 3
            identical += 1
    print(identical)

    float bound = 1.0 / math.sqrt(float(3))
    int outside = 0
    for unit in range(4)
        for feature in range(3)
            if math.abs(float(layer.weight.raw()[unit, feature].item())) > bound
                outside += 1
    print(outside)

    dnn.LinearLayer repeated = try dnn.Linear(features_in = 3, features_out = 4)
    dnn.LinearLayer reseeded = try dnn.Linear(features_in = 3, features_out = 4, seed = 7)
    print(repeated.weight.raw()[0, 0].item() == layer.weight.raw()[0, 0].item())
    print(reseeded.weight.raw()[0, 0].item() != layer.weight.raw()[0, 0].item())

    dnn.Conv2DLayer convolution = try dnn.Conv2D(
        channels_in = 2, channels_out = 3, kernel = 3
    )
    int same_filters = 0
    for filter_index in range(1, 3)
        if convolution.weight.raw()[filter_index, 0, 0, 0].item() == convolution.weight.raw()[0, 0, 0, 0].item()
            same_filters += 1
    print(same_filters)
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

initialization_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/initialization.qui")"
initialization_expected="$(printf '4\n3\n0\n0\ntrue\ntrue\n0')"
if [[ "$initialization_output" != "$initialization_expected" ]]; then
    echo "unexpected dnn initialization output: $initialization_output" >&2
    exit 1
fi

echo "dnn integration: ok"
