#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Effective only with the core's test-only fake GPU backend.
export QUIDRA_TEST_FAKE_GPU_COUNT=2

cat > "$TMP/device-check.qui" <<'QUI'
import dnn

int | error compile_device_surface()
    dnn.LinearLayer layer = try dnn.Linear(features_in = 2, features_out = 1)
    tensor<float32> direct = tensor.zeros<float32>([1, 2], gpu = 0)
    tensor<float32> transferred = tensor.ones<float32>([1, 2]).gpu(0)
    tensor<float32> roundtrip = transferred.cpu()
    tensor<float32> output = layer.forward(direct)
    print(roundtrip.shape()[1])
    print(output.shape()[1])
    return 0
QUI

QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/device-check.qui" >/dev/null

cat > "$TMP/no-fallback-create.qui" <<'QUI'
tensor<float32> value = tensor.zeros<float32>([1], gpu = 2147483647)
print(value.shape()[0])
QUI

set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/no-fallback-create.qui" >"$TMP/create.out" 2>"$TMP/create.err"
status=$?
set -e
if [[ $status -eq 0 ]]; then
    echo "GPU creation unexpectedly fell back to CPU" >&2
    exit 1
fi
if ! grep -Fq "gpu(2147483647) is not available" "$TMP/create.err"; then
    echo "missing explicit unavailable-GPU diagnostic" >&2
    cat "$TMP/create.err" >&2
    exit 1
fi

cat > "$TMP/no-fallback-transfer.qui" <<'QUI'
tensor<float32> cpu = tensor.ones<float32>([1])
tensor<float32> value = cpu.gpu(2147483647)
print(value.shape()[0])
QUI

set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/no-fallback-transfer.qui" >"$TMP/transfer.out" 2>"$TMP/transfer.err"
status=$?
set -e
if [[ $status -eq 0 ]]; then
    echo "GPU transfer unexpectedly fell back to CPU" >&2
    exit 1
fi
if ! grep -Fq "gpu(2147483647) is not available" "$TMP/transfer.err"; then
    echo "missing explicit unavailable-GPU transfer diagnostic" >&2
    cat "$TMP/transfer.err" >&2
    exit 1
fi

expect_device_failure() {
    local source="$1"
    local expected="$2"
    set +e
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$source" >"$source.out" 2>"$source.err"
    local status=$?
    set -e
    if [[ $status -ne 101 ]]; then
        echo "expected runtime status 101 for $source, got $status" >&2
        cat "$source.out" >&2 || true
        cat "$source.err" >&2 || true
        exit 1
    fi
    if ! grep -Fq "$expected" "$source.err"; then
        echo "missing device diagnostic '$expected'" >&2
        cat "$source.err" >&2
        exit 1
    fi
    if [[ -s "$source.out" ]]; then
        echo "DNN produced CPU output before GPU failure" >&2
        cat "$source.out" >&2
        exit 1
    fi
}

cat > "$TMP/parameter-device-mismatch.qui" <<'QUI'
import dnn

int | error run()
    dnn.LinearLayer layer = try dnn.Linear(features_in = 2, features_out = 1)
    tensor<float32> samples_gpu = tensor.ones<float32>([1, 2], gpu = 0)
    tensor<float32> output = layer.forward(samples_gpu)
    print(output.shape()[1])
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI
expect_device_failure "$TMP/parameter-device-mismatch.qui" "input and Parameter/state tensors must be on the same device"

cat > "$TMP/same-device-compute.qui" <<'QUI'
import dnn

dnn.LinearLayer layer = dnn.LinearLayer(
    weight = neural.Parameter<float32>(
        value = tensor.ones<float32>([1, 2], gpu = 0)
    ),
    bias = neural.Parameter<float32>(
        value = tensor.zeros<float32>([1], gpu = 0)
    )
)
tensor<float32> samples = tensor.ones<float32>([1, 2], gpu = 0)
tensor<float32> output = layer.forward(samples)
print(output.shape()[0])
print(output.shape()[1])
print(output[0, 0].item())

tensor<float32> values = tensor.zeros<float32>([1, 2], gpu = 0)
values[0, 0] = float32(-1)
values[0, 1] = float32(1)
tensor<float32> activated = dnn.relu(values)
tensor<float32> probabilities = dnn.softmax(values)
print(activated[0, 0].item())
print(activated[0, 1].item())
float32 probability_total = probabilities[0, 0].item() + probabilities[0, 1].item()
print(probability_total > float32(0.9999) and probability_total < float32(1.0001))

dnn.BatchNormLayer normalization = dnn.BatchNormLayer(
    scale = neural.Parameter<float32>(
        value = tensor.ones<float32>([2], gpu = 0)
    ),
    bias = neural.Parameter<float32>(
        value = tensor.zeros<float32>([2], gpu = 0)
    ),
    running_mean = neural.State<tensor<float32>>(
        value = tensor.zeros<float32>([2], gpu = 0)
    ),
    running_variance = neural.State<tensor<float32>>(
        value = tensor.ones<float32>([2], gpu = 0)
    ),
    momentum = 0.1,
    epsilon = 0.00001
)
tensor<float32> normalized = normalization.infer(
    tensor.ones<float32>([1, 2], gpu = 0)
)
print(normalized.shape()[1])
print(normalized[0, 0].item() > float32(0.9))
QUI

compute_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/same-device-compute.qui")"
compute_expected="$(printf '1\n1\n2.0\n0.0\n1.0\ntrue\n2\ntrue')"
if [[ "$compute_output" != "$compute_expected" ]]; then
    echo "unexpected same-GPU DNN output:" >&2
    printf '%s\n' "$compute_output" >&2
    exit 1
fi

cat > "$TMP/convolution-gpu.qui" <<'QUI'
import dnn

tensor<float32> kernel = tensor.zeros<float32>([1, 1, 1, 1], gpu = 0)
kernel[0, 0, 0, 0] = float32(2)
dnn.Conv2DLayer convolution = dnn.Conv2DLayer(
    weight = neural.Parameter<float32>(value = kernel),
    bias = neural.Parameter<float32>(
        value = tensor.zeros<float32>([1], gpu = 0)
    ),
    stride = 1,
    padding = 0
)
tensor<float32> pixels = tensor.ones<float32>([1, 1, 2, 2], gpu = 0)
tensor<float32> filtered = convolution.forward(pixels)
print(filtered.shape()[2])
print(filtered.shape()[3])
print(filtered[0, 0, 1, 1].item())
QUI
conv_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/convolution-gpu.qui")"
conv_expected="$(printf '2\n2\n2.0')"
if [[ "$conv_output" != "$conv_expected" ]]; then
    echo "unexpected same-GPU convolution output:" >&2
    printf '%s\n' "$conv_output" >&2
    exit 1
fi

cat > "$TMP/tracked-gpu.qui" <<'QUI'
import dnn

class Model
    dnn.LinearLayer dense

int | error run()
    dnn.LinearLayer layer = dnn.LinearLayer(
        weight = neural.Parameter<float32>(
            value = tensor.ones<float32>([1, 2], gpu = 0)
        ),
        bias = neural.Parameter<float32>(
            value = tensor.zeros<float32>([1], gpu = 0)
        )
    )
    Model model = Model(dense = layer)
    tensor<float32> samples = tensor.ones<float32>([1, 2], gpu = 0)
    tensor<float32> target = tensor.zeros<float32>([1, 1], gpu = 0)

    neural<float32> tracked = neural.track(samples)
    neural<float32> prediction = model.dense.forward(tracked)
    neural<float32> loss = dnn.mse(prediction, target)
    print(prediction.untrack()[0, 0].item())
    print(loss.untrack().item())

    neural.Gradients gradients = neural.grad(loss)
    float32 before = model.dense.weight.raw()[0, 0].item()
    dnn.SGDOptimizer optimizer = try dnn.SGD(rate = 0.1)
    optimizer.step(&model, gradients)
    float32 after = model.dense.weight.raw()[0, 0].item()
    print(after < before)
    print(after > float32(0.59) and after < float32(0.61))
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

tracked_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/tracked-gpu.qui")"
tracked_expected="$(printf '2.0\n4.0\ntrue\ntrue')"
if [[ "$tracked_output" != "$tracked_expected" ]]; then
    echo "unexpected tracked GPU DNN output:" >&2
    printf '%s\n' "$tracked_output" >&2
    exit 1
fi



cat > "$TMP/training-gpu.qui" <<'QUI'
import dnn

class ConvModel
    dnn.Conv2DLayer convolution

class LinearModel
    dnn.LinearLayer dense

class ParameterModel
    neural.Parameter<float32> value

int | error run()
    tensor<float32> kernel = tensor.ones<float32>([1, 1, 1, 1], gpu = 0)
    dnn.Conv2DLayer convolution = dnn.Conv2DLayer(
        weight = neural.Parameter<float32>(value = kernel),
        bias = neural.Parameter<float32>(
            value = tensor.zeros<float32>([1], gpu = 0)
        ),
        stride = 1,
        padding = 0
    )
    ConvModel conv_model = ConvModel(convolution = convolution)
    tensor<float32> pixels = tensor.ones<float32>([1, 1, 2, 2], gpu = 0)
    tensor<float32> zero_image = tensor.zeros<float32>([1, 1, 2, 2], gpu = 0)
    neural<float32> conv_prediction = conv_model.convolution.forward(neural.track(pixels))
    neural.Gradients conv_gradients = neural.grad(dnn.mse(conv_prediction, zero_image))
    float32 conv_before = conv_model.convolution.weight.raw()[0, 0, 0, 0].item()
    dnn.SGDOptimizer sgd = dnn.SGDOptimizer(rate = 0.1)
    sgd.step(&conv_model, conv_gradients)
    print(conv_model.convolution.weight.raw()[0, 0, 0, 0].item() < conv_before)

    dnn.BatchNormLayer normalization = dnn.BatchNormLayer(
        scale = neural.Parameter<float32>(
            value = tensor.ones<float32>([2], gpu = 0)
        ),
        bias = neural.Parameter<float32>(
            value = tensor.zeros<float32>([2], gpu = 0)
        ),
        running_mean = neural.State<tensor<float32>>(
            value = tensor.zeros<float32>([2], gpu = 0)
        ),
        running_variance = neural.State<tensor<float32>>(
            value = tensor.ones<float32>([2], gpu = 0)
        ),
        momentum = 0.1,
        epsilon = 0.00001
    )
    tensor<float32> norm_values = tensor.ones<float32>([2, 2], gpu = 0)
    neural<float32> normalized = normalization.forward(neural.track(norm_values))
    neural.Gradients norm_gradients = neural.grad(neural.mean(normalized * normalized))
    print(normalized.untrack().shape()[1])
    print(normalization.running_mean.value[0].item() > float32(0))

    dnn.DropoutLayer dropout = dnn.DropoutLayer(
        rate = 0.5,
        rng = neural.State<uint64>(value = uint64(17))
    )
    neural<float32> dropped = dropout.forward(neural.track(norm_values))
    neural.Gradients dropout_gradients = neural.grad(neural.mean(dropped * dropped))
    print(dropped.untrack().shape()[0])
    print(dropout.rng.value != uint64(17))

    ParameterModel cpu_dropout_model = ParameterModel(
        value = neural.Parameter<float32>(
            value = tensor.ones<float32>([2, 2])
        )
    )
    ParameterModel gpu_dropout_model = ParameterModel(
        value = neural.Parameter<float32>(
            value = tensor.ones<float32>([2, 2], gpu = 0)
        )
    )
    dnn.DropoutLayer cpu_dropout_backward = dnn.DropoutLayer(
        rate = 0.5,
        rng = neural.State<uint64>(value = uint64(29))
    )
    dnn.DropoutLayer gpu_dropout_backward = dnn.DropoutLayer(
        rate = 0.5,
        rng = neural.State<uint64>(value = uint64(29))
    )
    neural<float32> cpu_dropout_output = cpu_dropout_backward.forward(
        cpu_dropout_model.value.track()
    )
    neural<float32> gpu_dropout_output = gpu_dropout_backward.forward(
        gpu_dropout_model.value.track()
    )
    neural.Gradients cpu_dropout_gradients = neural.grad(neural.mean(cpu_dropout_output))
    neural.Gradients gpu_dropout_gradients = neural.grad(neural.mean(gpu_dropout_output))
    dnn.SGDOptimizer cpu_dropout_sgd = dnn.SGDOptimizer(rate = 0.1)
    dnn.SGDOptimizer gpu_dropout_sgd = dnn.SGDOptimizer(rate = 0.1)
    cpu_dropout_sgd.step(&cpu_dropout_model, cpu_dropout_gradients)
    gpu_dropout_sgd.step(&gpu_dropout_model, gpu_dropout_gradients)
    tensor<float32> gpu_dropout_weight = gpu_dropout_model.value.raw().cpu()
    print(
        cpu_dropout_model.value.raw()[0, 0].item() == gpu_dropout_weight[0, 0].item()
        and cpu_dropout_model.value.raw()[0, 1].item() == gpu_dropout_weight[0, 1].item()
        and cpu_dropout_model.value.raw()[1, 0].item() == gpu_dropout_weight[1, 0].item()
        and cpu_dropout_model.value.raw()[1, 1].item() == gpu_dropout_weight[1, 1].item()
    )

    ParameterModel accumulation_model = ParameterModel(
        value = neural.Parameter<float32>(
            value = tensor.ones<float32>([1], gpu = 0)
        )
    )
    neural<float32> accumulation_value = accumulation_model.value.track()
    neural.Gradients accumulation_gradients = neural.grad(
        neural.mean(accumulation_value + accumulation_value)
    )
    dnn.SGDOptimizer accumulation_sgd = dnn.SGDOptimizer(rate = 0.1)
    accumulation_sgd.step(&accumulation_model, accumulation_gradients)
    float32 accumulation_after = accumulation_model.value.raw()[0].item()
    print(
        accumulation_after > float32(0.7999)
        and accumulation_after < float32(0.8001)
    )

    dnn.LinearLayer dense = dnn.LinearLayer(
        weight = neural.Parameter<float32>(
            value = tensor.ones<float32>([1, 1], gpu = 0)
        ),
        bias = neural.Parameter<float32>(
            value = tensor.zeros<float32>([1], gpu = 0)
        )
    )
    LinearModel model = LinearModel(dense = dense)
    dnn.AdamOptimizer adam = dnn.AdamOptimizer(
        rate = 0.1,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 0.00000001,
        iteration = neural.State<int>(value = 0),
        moments = neural.State<bin>(value = bin.fill(0, 0))
    )
    tensor<float32> sample = tensor.ones<float32>([1, 1], gpu = 0)
    tensor<float32> target = tensor.zeros<float32>([1, 1], gpu = 0)
    float32 before = model.dense.weight.raw()[0, 0].item()
    neural<float32> prediction = model.dense.forward(neural.track(sample))
    neural.Gradients gradients = neural.grad(dnn.mse(prediction, target))
    adam.step(&model, gradients)
    print(adam.iteration.value)
    float32 first_after = model.dense.weight.raw()[0, 0].item()
    print(first_after < before)
    neural<float32> second_prediction = model.dense.forward(neural.track(sample))
    neural.Gradients second_gradients = neural.grad(dnn.mse(second_prediction, target))
    adam.step(&model, second_gradients)
    print(adam.iteration.value)
    print(model.dense.weight.raw()[0, 0].item() < first_after)
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

training_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/training-gpu.qui")"
training_expected="$(printf 'true\n2\ntrue\n2\ntrue\ntrue\ntrue\n1\ntrue\n2\ntrue')"
if [[ "$training_output" != "$training_expected" ]]; then
    echo "unexpected GPU training output:" >&2
    printf '%s\n' "$training_output" >&2
    exit 1
fi

cat > "$TMP/all-reduce-gpu.qui" <<'QUI'
import dnn

tensor<float32> first = tensor.ones<float32>([2], gpu = 0)
tensor<float32> second = tensor.ones<float32>([2], gpu = 1) * float32(2)
tensor<float32>[] values = [first, second]
dnn.all_reduce(&values)
tensor<float32> first_cpu = values[0].cpu()
tensor<float32> second_cpu = values[1].cpu()
print(first_cpu[0].item())
print(first_cpu[1].item())
print(second_cpu[0].item())
print(second_cpu[1].item())
QUI
all_reduce_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/all-reduce-gpu.qui")"
all_reduce_expected="$(printf '3.0\n3.0\n3.0\n3.0')"
if [[ "$all_reduce_output" != "$all_reduce_expected" ]]; then
    echo "unexpected fake-GPU all-reduce output:" >&2
    printf '%s\n' "$all_reduce_output" >&2
    exit 1
fi

equivalence_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$REPOSITORY_ROOT/tests/fake_gpu_equivalence.qui")"
equivalence_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$equivalence_output" != "$equivalence_expected" ]]; then
    echo "fake GPU CPU-equivalence regression failed:" >&2
    printf '%s\n' "$equivalence_output" >&2
    exit 1
fi

echo "dnn device contracts: ok"
