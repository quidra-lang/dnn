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
    dnn.FC dense

int | error run()
    dnn.FC dense = try dnn.FC(features_in = 2, features_out = 1)
    Model model
    model.dense = dense
    tensor<float32> samples = tensor.ones<float32>([1, 2])
    tensor<float32> targets = tensor.zeros<float32>([1, 1])
    tensor<float32> prediction = dnn.relu(
        model.dense.forward(samples.track())
    )
    tensor<float32> loss = dnn.mse(prediction, targets)
    dnn.SGD optimizer = try dnn.SGD(rate = 0.1)
    optimizer.zero_grad(&model)
    loss.backward(&model)
    optimizer.step(&model)

    print(prediction.shape()[0])
    print(prediction.shape()[1])

    float32 first = loss.item()
    float32 latest = first
    for iteration in range(20)
        tensor<float32> step_prediction = model.dense.forward(
            samples.track()
        )
        tensor<float32> step_loss = dnn.mse(step_prediction, targets)
        optimizer.zero_grad(&model)
        step_loss.backward(&model)
        optimizer.step(&model)
        latest = step_loss.item()
    print(latest < first)
    return 0

auto | error result = run()
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
    dnn.Conv2D convolution = try dnn.Conv2D(
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
    tensor<float32> logits = values.track()
    print(dnn.cross_entropy(logits, target).item() > float32(0))
    print(dnn.binary_cross_entropy(dnn.sigmoid(logits), target).item() > float32(0))
    return 0

auto | error result = run()
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
    dnn.BatchNorm normalization = try dnn.BatchNorm(features = 2)
    tensor<float32> values = tensor.ones<float32>([2, 2])
    tensor<float32> trained = normalization.forward(values.track())
    tensor<float32> inferred = normalization.infer(values)
    print(trained.shape()[1])
    print(inferred.shape()[1])
    print(inferred[0, 0].item() < float32(1))

    dnn.Dropout masking = try dnn.Dropout(rate = 0.5, seed = 17)
    tensor<float32> masked = masking.forward(values.track())
    print(masked.shape()[0])
    print(masked.shape()[1] == 2)
    print(masked.is_tracked())
    print(masking.infer(values)[0, 0].item())

    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

stateful_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/stateful-layers.qui")"
stateful_expected="$(printf '2\n2\ntrue\n2\ntrue\ntrue\n1.0')"
if [[ "$stateful_output" != "$stateful_expected" ]]; then
    echo "unexpected dnn stateful output: $stateful_output" >&2
    exit 1
fi

cat > "$TMP/batchnorm-generic-autograd.qui" <<'QUI'
import dnn

int | error run()
    dnn.BatchNorm normalization = try dnn.BatchNorm(features = 2)
    tensor<float32> values = tensor.zeros<float32>([2, 2])
    values[0, 0] = float32(1)
    values[0, 1] = float32(2)
    values[1, 0] = float32(3)
    values[1, 1] = float32(4)
    tensor<float32> tracked = values.track()
    tensor<float32> output = normalization.forward(tracked)
    (output * output).mean().backward(&normalization, &tracked, track = true)
    print(tracked.grad.is_tracked())
    print(normalization.scale.gradient().is_tracked())
    print(normalization.bias.gradient().is_tracked())
    print(not normalization.running_mean.raw().is_tracked())
    print(not normalization.running_variance.raw().is_tracked())
    tracked.grad.mean().backward(&tracked)
    print(tracked.grad.untrack().shape()[0] == 2)
    tensor<float32> inferred = normalization.infer(values.track())
    print(inferred.is_tracked())
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

batchnorm_generic_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/batchnorm-generic-autograd.qui")"
batchnorm_generic_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$batchnorm_generic_output" != "$batchnorm_generic_expected" ]]; then
    echo "unexpected generic BatchNorm/autograd output:" >&2
    printf '%s\n' "$batchnorm_generic_output" >&2
    exit 1
fi

cat > "$TMP/adam.qui" <<'QUI'
import dnn

class Model
    dnn.FC dense

int | error run()
    dnn.FC dense = try dnn.FC(features_in = 1, features_out = 1)
    Model model
    model.dense = dense
    dnn.Adam optimizer = try dnn.Adam(rate = 0.1)
    tensor<float32> samples = tensor.ones<float32>([1, 1])
    tensor<float32> targets = tensor.ones<float32>([1, 1])
    float32 before = model.dense.weight.raw()[0, 0].item()
    tensor<float32> prediction = model.dense.forward(samples.track())
    tensor<float32> loss = dnn.mse(prediction, targets)
    optimizer.zero_grad(&model)
    loss.backward(&model)
    optimizer.step(&model)
    print(model.dense.weight.raw()[0, 0].item() != before)
    print(model.dense.weight.raw()[0, 0].item() > before)
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

adam_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/adam.qui")"
if [[ "$adam_output" != "$(printf 'true\ntrue')" ]]; then
    echo "unexpected dnn Adam output: $adam_output" >&2
    exit 1
fi

cat > "$TMP/numeric-stability.qui" <<'QUI'
import dnn

class Model
    dnn.FC dense

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
    tensor<float32> saturated_loss = dnn.binary_cross_entropy(
        saturated.track(), labels
    )
    print(saturated_loss.item() < float32(0.001))

    tensor<float32> logits = tensor.zeros<float32>([1, 2])
    logits[0, 1] = float32(200.0)
    tensor<float32> target = tensor.zeros<float32>([1, 2])
    target[0, 0] = float32(1.0)
    print(dnn.cross_entropy(logits.track(), target).item())

    dnn.FC dense = try dnn.FC(features_in = 2, features_out = 2)
    Model model
    model.dense = dense
    tensor<float32> samples = tensor.ones<float32>([1, 2])
    tensor<float32> classes = tensor.zeros<float32>([1, 2])
    classes[0, 1] = float32(1)
    tensor<float32> scores = model.dense.forward(samples.track())
    tensor<float32> loss = dnn.cross_entropy(scores, classes)
    float32 before = model.dense.bias.raw()[1].item()
    dnn.SGD optimizer = try dnn.SGD(rate = 0.5)
    optimizer.zero_grad(&model)
    loss.backward(&model)
    optimizer.step(&model)
    print(model.dense.bias.raw()[1].item() > before)
    return 0

auto | error result = run()
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

cat > "$TMP/conv-generic-autograd.qui" <<'QUI'
import dnn

dnn.Conv2D convolution
convolution.weight = dnn.Parameter<float32>(
    value = tensor.ones<float32>([1, 1, 2, 2])
)
convolution.bias = dnn.Parameter<float32>(
    value = tensor.zeros<float32>([1])
)
convolution.step = 1
convolution.border = 0

tensor<float32> input = tensor.ones<float32>([1, 1, 3, 3]).track()
tensor<float32> output = convolution.forward(input)
print(output.shape()[0] == 1)
print(output.shape()[1] == 1)
print(output.shape()[2] == 2)
print(output.shape()[3] == 2)
print(output.untrack()[0, 0, 0, 0].item() == float32(4))

output.mean().backward(&convolution, &input, track = true)
print(input.grad.is_tracked())
print(convolution.weight.gradient().is_tracked())
print(convolution.bias.gradient().is_tracked())
tensor<float32> input_grad = input.grad.untrack()
tensor<float32> weight_grad = convolution.weight.gradient().untrack()
tensor<float32> bias_grad = convolution.bias.gradient().untrack()
print(input_grad[0, 0, 0, 0].item() == float32(0.25))
print(input_grad[0, 0, 1, 1].item() == float32(1))
print(weight_grad[0, 0, 0, 0].item() == float32(1))
print(bias_grad[0].item() == float32(1))
input.grad.mean().backward(&input)
print(input.grad.untrack().shape()[2] == 3)

dnn.Conv2D padded
padded.weight = dnn.Parameter<float32>(
    value = tensor.ones<float32>([1, 1, 3, 3])
)
padded.bias = dnn.Parameter<float32>(
    value = tensor.zeros<float32>([1])
)
padded.step = 2
padded.border = 1
tensor<float32> padded_output = padded.forward(
    tensor.ones<float32>([1, 1, 4, 5])
)
print(padded_output.shape()[2] == 2)
print(padded_output.shape()[3] == 3)

dnn.Conv2D asymmetric
tensor<float32> asymmetric_weight = tensor.zeros<float32>([1, 1, 2, 2])
asymmetric_weight[0, 0, 0, 0] = float32(10)
asymmetric_weight[0, 0, 0, 1] = float32(1)
asymmetric.weight = dnn.Parameter<float32>(value = asymmetric_weight)
asymmetric.bias = dnn.Parameter<float32>(
    value = tensor.zeros<float32>([1])
)
asymmetric.step = 1
asymmetric.border = 0
tensor<float32> asymmetric_input = tensor.zeros<float32>([1, 1, 3, 3])
asymmetric_input[0, 0, 0, 0] = float32(1)
asymmetric_input[0, 0, 0, 1] = float32(2)
asymmetric_input[0, 0, 0, 2] = float32(3)
asymmetric_input[0, 0, 1, 0] = float32(4)
asymmetric_input[0, 0, 1, 1] = float32(5)
asymmetric_input[0, 0, 1, 2] = float32(6)
asymmetric_input[0, 0, 2, 0] = float32(7)
asymmetric_input[0, 0, 2, 1] = float32(8)
asymmetric_input[0, 0, 2, 2] = float32(9)
tensor<float32> asymmetric_output = asymmetric.forward(asymmetric_input)
print(asymmetric_output[0, 0, 0, 0].item() == float32(12))
print(asymmetric_output[0, 0, 1, 1].item() == float32(56))
QUI
conv_generic_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/conv-generic-autograd.qui")"
conv_generic_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$conv_generic_output" != "$conv_generic_expected" ]]; then
    echo "unexpected generic Conv2D/autograd output:" >&2
    printf '%s\n' "$conv_generic_output" >&2
    exit 1
fi

cat > "$TMP/reference-primitives.qui" <<'QUI'
import dnn

dnn.Conv2D grouped
tensor<float32> grouped_weight = tensor.zeros<float32>([4, 2, 1, 1])
grouped_weight[0, 0, 0, 0] = float32(1)
grouped_weight[1, 1, 0, 0] = float32(1)
grouped_weight[2, 0, 0, 0] = float32(1)
grouped_weight[3, 1, 0, 0] = float32(1)
grouped.weight = dnn.Parameter<float32>(value = grouped_weight)
grouped.bias = dnn.Parameter<float32>(
    value = tensor.zeros<float32>([4])
)
grouped.step = 1
grouped.border = 0
grouped.group_count = 2

tensor<float32> grouped_input = tensor.zeros<float32>([1, 4, 1, 1])
grouped_input[0, 0, 0, 0] = float32(1)
grouped_input[0, 1, 0, 0] = float32(2)
grouped_input[0, 2, 0, 0] = float32(3)
grouped_input[0, 3, 0, 0] = float32(4)
tensor<float32> grouped_output = grouped.forward(grouped_input)
print(grouped_output[0, 0, 0, 0].item() == float32(1))
print(grouped_output[0, 1, 0, 0].item() == float32(2))
print(grouped_output[0, 2, 0, 0].item() == float32(3))
print(grouped_output[0, 3, 0, 0].item() == float32(4))

tensor<float32> pool_input = tensor.zeros<float32>([1, 1, 3, 3])
for y in range(3)
    for x in range(3)
        pool_input[0, 0, y, x] = float32(y * 3 + x + 1)
tensor<float32> pool_tracked = pool_input.track()
tensor<float32> pool_output = dnn.max_pool2d(pool_tracked, 2, 1)
print(pool_output.shape()[2] == 2)
print(pool_output.shape()[3] == 2)
tensor<float32> pool_values = pool_output.untrack()
print(pool_values[0, 0, 0, 0].item() == float32(5))
print(pool_values[0, 0, 0, 1].item() == float32(6))
print(pool_values[0, 0, 1, 0].item() == float32(8))
print(pool_values[0, 0, 1, 1].item() == float32(9))
pool_output.mean().backward(&pool_tracked)
tensor<float32> pool_gradient = pool_tracked.grad.untrack()
print(pool_gradient[0, 0, 1, 1].item() == float32(0.25))
print(pool_gradient[0, 0, 0, 0].item() == float32(0))

tensor<float32> average_input = tensor.zeros<float32>([1, 1, 2, 2])
average_input[0, 0, 0, 0] = float32(1)
average_input[0, 0, 0, 1] = float32(2)
average_input[0, 0, 1, 0] = float32(3)
average_input[0, 0, 1, 1] = float32(4)
tensor<float32> average_tracked = average_input.track()
tensor<float32> average_output = dnn.global_average_pool2d(average_tracked)
print(average_output.shape()[0] == 1)
print(average_output.shape()[1] == 1)
tensor<float32> average_values = average_output.untrack()
print(average_values[0, 0].item() == float32(2.5))
average_output.mean().backward(&average_tracked)
print(average_tracked.grad.untrack()[0, 0, 0, 0].item() == float32(0.25))

tensor<float32> lrn_input = tensor.zeros<float32>([1, 3, 1, 1])
lrn_input[0, 0, 0, 0] = float32(1)
lrn_input[0, 1, 0, 0] = float32(2)
lrn_input[0, 2, 0, 0] = float32(3)
tensor<float32> lrn_tracked = lrn_input.track()
tensor<float32> lrn_output = dnn.local_response_normalization(
    lrn_tracked, size = 3, alpha = 1.0, beta = 1.0, k = 1.0
)
tensor<float32> lrn_values = lrn_output.untrack()
float32 lrn0 = lrn_values[0, 0, 0, 0].item()
float32 lrn1 = lrn_values[0, 1, 0, 0].item()
float32 lrn2 = lrn_values[0, 2, 0, 0].item()
print(math.abs(lrn0 - float32(1.0 / 6.0)) < float32(0.0001))
print(math.abs(lrn1 - float32(2.0 / 15.0)) < float32(0.0001))
print(math.abs(lrn2 - float32(3.0 / 14.0)) < float32(0.0001))
lrn_output.mean().backward(&lrn_tracked)
print(lrn_tracked.grad.shape()[1] == 3)

tensor<float32> shortcut_input = tensor.zeros<float32>([1, 2, 2, 2])
shortcut_input[0, 0, 0, 0] = float32(1)
shortcut_input[0, 1, 0, 0] = float32(2)
tensor<float32> shortcut_tracked = shortcut_input.track()
dnn.model.ResNetBasicOptionABlock shortcut_block = dnn.model.ResNetBasicOptionABlock(
    channels_in = 2, channels_out = 4, stride = 2
)
shortcut_block.conv1.weight.replace(tensor.zeros<float32>([4, 2, 3, 3]))
shortcut_block.conv1.bias.replace(tensor.zeros<float32>([4]))
shortcut_block.conv2.weight.replace(tensor.zeros<float32>([4, 4, 3, 3]))
shortcut_block.conv2.bias.replace(tensor.zeros<float32>([4]))
tensor<float32> shortcut = shortcut_block.infer(shortcut_tracked)
print(shortcut.shape()[1] == 4)
print(shortcut.shape()[2] == 1)
print(shortcut.shape()[3] == 1)
tensor<float32> shortcut_values = shortcut.untrack()
print(shortcut_values[0, 0, 0, 0].item() == float32(1))
print(shortcut_values[0, 1, 0, 0].item() == float32(2))
print(shortcut_values[0, 2, 0, 0].item() == float32(0))
print(shortcut_values[0, 3, 0, 0].item() == float32(0))
shortcut.mean().backward(&shortcut_tracked)
tensor<float32> shortcut_gradient = shortcut_tracked.grad.untrack()
print(shortcut_gradient[0, 0, 0, 0].item() == float32(0.25))
print(shortcut_gradient[0, 0, 0, 1].item() == float32(0))
QUI
reference_primitives_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/reference-primitives.qui")"
reference_primitives_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$reference_primitives_output" != "$reference_primitives_expected" ]]; then
    echo "unexpected reference CNN primitive output:" >&2
    printf '%s\n' "$reference_primitives_output" >&2
    exit 1
fi

cat > "$TMP/initialization.qui" <<'QUI'
import dnn

int | error run()
    dnn.FC layer = try dnn.FC(features_in = 3, features_out = 4)
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

    dnn.FC repeated = try dnn.FC(features_in = 3, features_out = 4)
    dnn.FC reseeded = try dnn.FC(features_in = 3, features_out = 4, seed = 7)
    print(repeated.weight.raw()[0, 0].item() == layer.weight.raw()[0, 0].item())
    print(reseeded.weight.raw()[0, 0].item() != layer.weight.raw()[0, 0].item())

    dnn.Conv2D convolution = try dnn.Conv2D(
        channels_in = 2, channels_out = 3, kernel = 3
    )
    int same_filters = 0
    for filter_index in range(1, 3)
        if convolution.weight.raw()[filter_index, 0, 0, 0].item() == convolution.weight.raw()[0, 0, 0, 0].item()
            same_filters += 1
    print(same_filters)
    return 0

auto | error result = run()
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


cat > "$TMP/tracked-forward-gradients.qui" <<'QUI'
import dnn

class GradientModel
    dnn.FC dense

dnn.FC dense = dnn.FC(features_in = 2, features_out = 1)
GradientModel model
model.dense = dense
tensor<float32> samples = tensor.ones<float32>([1, 2]).track()
tensor<float32> prediction = model.dense.forward(samples)
prediction.mean().backward(&model, &samples)
print(samples.grad.shape()[0])
print(samples.grad.shape()[1])
print(model.dense.weight.gradient()[0, 0].item())
print(model.dense.weight.gradient()[0, 1].item())
print(model.dense.bias.gradient()[0].item())
QUI

tracked_forward_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/tracked-forward-gradients.qui")"
tracked_forward_expected="$(printf '1\n2\n1.0\n1.0\n1.0')"
if [[ "$tracked_forward_output" != "$tracked_forward_expected" ]]; then
    echo "unexpected DNN tracked input/Parameter gradients:" >&2
    printf '%s\n' "$tracked_forward_output" >&2
    exit 1
fi

cat > "$TMP/linear-leading-and-higher-order.qui" <<'QUI'
import dnn

class HigherOrderModel
    dnn.FC dense

dnn.FC dense = dnn.FC(features_in = 2, features_out = 1)
HigherOrderModel model
model.dense = dense
tensor<float32> samples = tensor.ones<float32>([2, 3, 2]).track()
tensor<float32> prediction = model.dense.forward(samples)
print(prediction.shape()[0] == 2)
print(prediction.shape()[1] == 3)
print(prediction.shape()[2] == 1)
(prediction * prediction).mean().backward(&model, &samples, track = true)
print(samples.grad.is_tracked())
print(model.dense.weight.gradient().is_tracked())
print(model.dense.bias.gradient().is_tracked())
samples.grad.mean().backward(&samples)
print(samples.grad.untrack().shape()[0] == 2)
QUI

linear_higher_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/linear-leading-and-higher-order.qui")"
linear_higher_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$linear_higher_output" != "$linear_higher_expected" ]]; then
    echo "unexpected DNN FC leading-shape/higher-order output:" >&2
    printf '%s\n' "$linear_higher_output" >&2
    exit 1
fi

cat > "$TMP/untracked-forward-no-graph.qui" <<'QUI'
import dnn

class InferenceModel
    dnn.FC dense

dnn.FC dense = dnn.FC(features_in = 2, features_out = 1)
InferenceModel model
model.dense = dense
tensor<float32> samples = tensor.ones<float32>([1, 2])
tensor<float32> prediction = model.dense.forward(samples)
prediction.mean().backward(&model)
QUI

set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/untracked-forward-no-graph.qui" >"$TMP/untracked-forward-no-graph.out" 2>"$TMP/untracked-forward-no-graph.err"
untracked_forward_rc=$?
set -e
if [[ "$untracked_forward_rc" -ne 101 ]]; then
    echo "DNN untracked forward unexpectedly created an autograd graph" >&2
    exit 1
fi
grep -Fq "backward() requires a tracked tensor" "$TMP/untracked-forward-no-graph.err"

cat > "$TMP/untracked-forward-no-parameter-grad.qui" <<'QUI'
import dnn

class InferenceModel
    dnn.FC dense

dnn.FC dense = dnn.FC(features_in = 2, features_out = 1)
InferenceModel model
model.dense = dense
tensor<float32> samples = tensor.ones<float32>([1, 2])
tensor<float32> prediction = model.dense.forward(samples)
print(prediction[0, 0].item())
print(model.dense.weight.gradient()[0, 0].item())
QUI

set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/untracked-forward-no-parameter-grad.qui" >"$TMP/untracked-forward-no-parameter-grad.out" 2>"$TMP/untracked-forward-no-parameter-grad.err"
untracked_parameter_rc=$?
set -e
if [[ "$untracked_parameter_rc" -ne 101 ]]; then
    echo "DNN Parameter unexpectedly acquired a gradient during untracked inference" >&2
    exit 1
fi
grep -Fq "autograd target gradient is not available" "$TMP/untracked-forward-no-parameter-grad.err"

cat > "$TMP/parameter-abstraction.qui" <<'QUI'
import dnn

class ParameterModel
    dnn.Parameter<float32> weight

ParameterModel model
model.weight = dnn.Parameter<float32>(
    value = tensor.ones<float32>([1])
)
dnn.Parameter<float32>[] parameters = reflect.collect<dnn.Parameter<float32>>(model)
print(len(parameters) == 1)
dnn.Parameter<float32> parameter = parameters[0]
tensor<float32> tracked = parameter.track()
(tracked * tracked).mean().backward(&model)
print(model.weight.has_grad())
print(model.weight.gradient().untrack()[0].item() == float32(2))
model.weight.clear_grad()
print(not parameter.has_grad())
parameter.replace(tensor.ones<float32>([1]) * float32(3))
print(model.weight.raw()[0].item() == float32(3))

tensor<float32> source = tensor.ones<float32>([1]).track()
dnn.Parameter<float32> isolated = dnn.Parameter<float32>(value = source)
tensor<float32> isolated_tracked = isolated.track()
(isolated_tracked * isolated_tracked).mean().backward(&isolated)
print(isolated.has_grad())
print(not source.has_grad())
QUI

parameter_abstraction_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/parameter-abstraction.qui")"
parameter_abstraction_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$parameter_abstraction_output" != "$parameter_abstraction_expected" ]]; then
    echo "unexpected DNN Parameter abstraction output:" >&2
    printf '%s\n' "$parameter_abstraction_output" >&2
    exit 1
fi


cat > "$TMP/state-abstraction.qui" <<'QUI'
import dnn

class StatefulModel
    dnn.Parameter<float32> weight
    dnn.State<float32> running
    tensor<float32> cache
    float32 epsilon

StatefulModel model
model.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1]))
model.running = dnn.State<float32>(value = tensor.zeros<float32>([1]))
model.cache = tensor.ones<float32>([4])
model.epsilon = float32(0.001)

dnn.Parameter<float32>[] parameters = reflect.collect<dnn.Parameter<float32>>(model)
dnn.State<float32>[] states = reflect.collect<dnn.State<float32>>(model)
print(len(parameters) == 1)
print(len(states) == 1)
print(states[0].same(model.running))
tensor<float32> tracked = tensor.ones<float32>([1]).track()
model.running.replace(tracked)
print(not model.running.raw().is_tracked())
print(model.cache.shape()[0] == 4)
print(model.epsilon == float32(0.001))
QUI

state_abstraction_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/state-abstraction.qui")"
state_abstraction_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$state_abstraction_output" != "$state_abstraction_expected" ]]; then
    echo "unexpected DNN State abstraction output:" >&2
    printf '%s\n' "$state_abstraction_output" >&2
    exit 1
fi


cat > "$TMP/persistence.qui" <<QUI
import dnn

class SavedModel
    dnn.Parameter<float32> weight
    dnn.State<float32> running
    tensor<float32> cache
    float32 epsilon

SavedModel model
model.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1]) * float32(2))
model.running = dnn.State<float32>(value = tensor.ones<float32>([1]) * float32(3))
model.cache = tensor.ones<float32>([1]) * float32(7)
model.epsilon = float32(0.25)
dnn.save(model, path = "$TMP/model.dnn")
model.weight.replace(tensor.zeros<float32>([1]))
model.running.replace(tensor.zeros<float32>([1]))
model.cache = tensor.zeros<float32>([1])
model.epsilon = float32(9)
dnn.load(&model, path = "$TMP/model.dnn")
print(model.weight.raw()[0].item() == float32(2))
print(model.running.raw()[0].item() == float32(3))
print(model.cache[0].item() == float32(0))
print(model.epsilon == float32(9))
QUI

persistence_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/persistence.qui")"
persistence_expected="$(printf 'true\ntrue\ntrue\ntrue')"
if [[ "$persistence_output" != "$persistence_expected" ]]; then
    echo "unexpected DNN persistence output:" >&2
    printf '%s\n' "$persistence_output" >&2
    exit 1
fi


cat > "$TMP/adam-persistence.qui" <<QUI
import dnn

class AdamModel
    dnn.Parameter<float32> weight

AdamModel model
model.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1]))
dnn.Adam optimizer = dnn.Adam(rate = 0.01)
tensor<float32> tracked = model.weight.track()
(tracked * tracked).mean().backward(&model)
optimizer.step(&model)
optimizer.save(path = "$TMP/adam.dnn")

dnn.Adam restored = dnn.Adam(rate = 0.5)
restored.load(&model, path = "$TMP/adam.dnn")
model.weight.clear_grad()
tensor<float32> next = model.weight.track()
(next * next).mean().backward(&model)
float32 before = model.weight.raw()[0].item()
restored.step(&model)
print(model.weight.raw()[0].item() != before)
QUI

adam_persistence_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/adam-persistence.qui")"
if [[ "$adam_persistence_output" != "true" ]]; then
    echo "unexpected Adam persistence output: $adam_persistence_output" >&2
    exit 1
fi


cat > "$TMP/adam-resume.qui" <<QUI
import dnn

class ResumeModel
    dnn.FC dense

void train_once(ResumeModel &model, dnn.Adam &optimizer)
    tensor<float32> samples = tensor.ones<float32>([1, 1])
    tensor<float32> targets = tensor.zeros<float32>([1, 1])
    tensor<float32> prediction = model.dense.forward(samples.track())
    tensor<float32> loss = dnn.mse(prediction, targets)
    optimizer.zero_grad(&model)
    loss.backward(&model)
    optimizer.step(&model)

int | error run()
    ResumeModel continuous
    continuous.dense = try dnn.FC(features_in = 1, features_out = 1)
    ResumeModel resumed
    resumed.dense = try dnn.FC(features_in = 1, features_out = 1)
    dnn.Adam continuous_optimizer = try dnn.Adam(rate = 0.1)
    dnn.Adam resumed_optimizer = try dnn.Adam(rate = 0.1)

    train_once(&continuous, &continuous_optimizer)
    try dnn.save(continuous, path = "$TMP/adam-model.dnn")
    try continuous_optimizer.save(path = "$TMP/adam-state.dnn")

    train_once(&continuous, &continuous_optimizer)

    try dnn.load(&resumed, path = "$TMP/adam-model.dnn")
    try resumed_optimizer.load(&resumed, path = "$TMP/adam-state.dnn")
    train_once(&resumed, &resumed_optimizer)

    print(continuous.dense.weight.raw()[0, 0].item() == resumed.dense.weight.raw()[0, 0].item())
    print(continuous.dense.bias.raw()[0].item() == resumed.dense.bias.raw()[0].item())
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

adam_resume_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/adam-resume.qui")"
adam_resume_expected="$(printf 'true\ntrue')"
if [[ "$adam_resume_output" != "$adam_resume_expected" ]]; then
    echo "Adam resume persistence diverged from uninterrupted training:" >&2
    printf '%s\n' "$adam_resume_output" >&2
    exit 1
fi

cat > "$TMP/explicit-gradient-targets.qui" <<'QUI'
import dnn

class PairModel
    dnn.FC encoder
    dnn.FC decoder

int | error run()
    PairModel pair
    pair.encoder = try dnn.FC(features_in = 1, features_out = 1)
    pair.decoder = try dnn.FC(features_in = 1, features_out = 1, seed = 2)
    tensor<float32> input = tensor.ones<float32>([1, 1]).track()
    tensor<float32> encoded = pair.encoder.forward(input)
    tensor<float32> decoded = pair.decoder.forward(encoded)
    tensor<float32> loss = decoded.mean()

    loss.backward(&pair.encoder)
    print(pair.encoder.weight.has_grad())
    print(not pair.decoder.weight.has_grad())
    print(not input.has_grad())

    pair.encoder.weight.clear_grad()
    loss.backward(&pair.encoder, &pair.decoder, &input)
    print(pair.encoder.weight.has_grad())
    print(pair.decoder.weight.has_grad())
    print(input.has_grad())

    dnn.SGD clear_optimizer = try dnn.SGD(rate = 0.1)
    clear_optimizer.zero_grad(&pair)
    print(not pair.encoder.weight.has_grad())
    print(not pair.decoder.weight.has_grad())

    dnn.FC generator = try dnn.FC(features_in = 1, features_out = 1, seed = 3)
    dnn.FC discriminator = try dnn.FC(features_in = 1, features_out = 1, seed = 4)
    tensor<float32> gan_input = tensor.ones<float32>([1, 1]).track()
    tensor<float32> generated = generator.forward(gan_input)
    tensor<float32> judged = discriminator.forward(generated)
    tensor<float32> d_loss = judged.mean()
    d_loss.backward(&discriminator)
    print(discriminator.weight.has_grad())
    print(not generator.weight.has_grad())

    dnn.BatchNorm normalization = try dnn.BatchNorm(features = 1)
    tensor<float32> state_input = tensor.ones<float32>([2, 1]).track()
    tensor<float32> normalized = normalization.forward(state_input)
    tensor<float32> running_mean_after_forward = normalization.running_mean.raw()
    tensor<float32> running_variance_after_forward = normalization.running_variance.raw()
    tensor<float32> state_loss = (normalized * normalized).mean()
    dnn.SGD optimizer = try dnn.SGD(rate = 0.1)
    optimizer.zero_grad(&normalization)
    state_loss.backward(&normalization)
    optimizer.step(&normalization)
    print((normalization.running_mean.raw() == running_mean_after_forward).all())
    print((normalization.running_variance.raw() == running_variance_after_forward).all())
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

explicit_gradient_targets_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/explicit-gradient-targets.qui")"
explicit_gradient_targets_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$explicit_gradient_targets_output" != "$explicit_gradient_targets_expected" ]]; then
    echo "unexpected explicit DNN gradient-target output:" >&2
    printf '%s\n' "$explicit_gradient_targets_output" >&2
    exit 1
fi

echo "dnn integration: ok"
