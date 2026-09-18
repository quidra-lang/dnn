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
dnn.SGDOptimizer optimizer = dnn.SGDOptimizer(rate = 0.1)
optimizer.step(&model, gradients)
float32 after = model.dense.weight.raw()[0, 0].item()
print(after < before)
print(after)
QUI

tracked_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/tracked-gpu.qui")"
tracked_expected="$(printf '2.0\n4.0\ntrue\n0.6')"
if [[ "$tracked_output" != "$tracked_expected" ]]; then
    echo "unexpected tracked GPU DNN output:" >&2
    printf '%s\n' "$tracked_output" >&2
    exit 1
fi

echo "dnn device contracts: ok"
