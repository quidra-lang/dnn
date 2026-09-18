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
    tensor<float32> input = tensor.ones<float32>([1, 2], gpu = 0)
    tensor<float32> output = layer.forward(input)
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

cat > "$TMP/same-device-unsupported.qui" <<'QUI'
import dnn

dnn.LinearLayer layer = dnn.LinearLayer(
    weight = neural.Parameter<float32>(
        value = tensor.ones<float32>([1, 2], gpu = 0)
    ),
    bias = neural.Parameter<float32>(
        value = tensor.zeros<float32>([1], gpu = 0)
    )
)
tensor<float32> input = tensor.ones<float32>([1, 2], gpu = 0)
tensor<float32> output = layer.forward(input)
print(output.shape()[1])
QUI
expect_device_failure "$TMP/same-device-unsupported.qui" "neural.affine is not supported on gpu(0) by the current neural backend"

cat > "$TMP/track-no-fallback.qui" <<'QUI'
tensor<float32> input = tensor.ones<float32>([1, 2], gpu = 0)
neural<float32> tracked = neural.track(input)
print(tracked.untrack().shape()[0])
QUI
expect_device_failure "$TMP/track-no-fallback.qui" "neural tensor conversion is not supported on gpu(0)"

echo "dnn device contracts: ok"
