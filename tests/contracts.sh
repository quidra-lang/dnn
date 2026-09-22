#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/contracts.qui" <<'QUI'
import dnn

dnn.mode.fast()
dnn.mode.deterministic()
dnn.mode.fast()

bool linear_rejected = false
dnn.Linear | error bad_linear = dnn.Linear(features_in = 2, features_out = 0)
match bad_linear
    dnn.Linear
        linear_rejected = false
    error
        linear_rejected = true
print(linear_rejected)

bool convolution_rejected = false
dnn.Conv2D | error bad_convolution = dnn.Conv2D(
    channels_in = 1,
    channels_out = 1,
    kernel = 3,
    stride = 0
)
match bad_convolution
    dnn.Conv2D
        convolution_rejected = false
    error
        convolution_rejected = true
print(convolution_rejected)

bool normalization_rejected = false
dnn.BatchNorm | error bad_normalization = dnn.BatchNorm(
    features = 2,
    momentum = 1.5
)
match bad_normalization
    dnn.BatchNorm
        normalization_rejected = false
    error
        normalization_rejected = true
print(normalization_rejected)

bool dropout_rejected = false
dnn.Dropout | error bad_dropout = dnn.Dropout(rate = 1.0)
match bad_dropout
    dnn.Dropout
        dropout_rejected = false
    error
        dropout_rejected = true
print(dropout_rejected)

bool sgd_rejected = false
dnn.SGD | error bad_sgd = dnn.SGD(rate = 0.0)
match bad_sgd
    dnn.SGD
        sgd_rejected = false
    error
        sgd_rejected = true
print(sgd_rejected)

bool adam_rejected = false
dnn.Adam | error bad_adam = dnn.Adam(beta1 = 1.0)
match bad_adam
    dnn.Adam
        adam_rejected = false
    error
        adam_rejected = true
print(adam_rejected)

float zero = 0.0
float nan_value = zero / zero
float infinity = 1.0 / zero

bool linear_overflow_rejected = false
dnn.Linear | error huge_linear = dnn.Linear(
    features_in = 3037000500,
    features_out = 3037000500
)
match huge_linear
    dnn.Linear
        linear_overflow_rejected = false
    error
        linear_overflow_rejected = true
print(linear_overflow_rejected)

bool convolution_overflow_rejected = false
dnn.Conv2D | error huge_convolution = dnn.Conv2D(
    channels_in = 1,
    channels_out = 1,
    kernel = 3037000500
)
match huge_convolution
    dnn.Conv2D
        convolution_overflow_rejected = false
    error
        convolution_overflow_rejected = true
print(convolution_overflow_rejected)

bool dropout_nan_rejected = false
dnn.Dropout | error nan_dropout = dnn.Dropout(rate = nan_value)
match nan_dropout
    dnn.Dropout
        dropout_nan_rejected = false
    error
        dropout_nan_rejected = true
print(dropout_nan_rejected)

bool sgd_infinity_rejected = false
dnn.SGD | error infinite_sgd = dnn.SGD(rate = infinity)
match infinite_sgd
    dnn.SGD
        sgd_infinity_rejected = false
    error
        sgd_infinity_rejected = true
print(sgd_infinity_rejected)

bool adam_nan_rejected = false
dnn.Adam | error nan_adam = dnn.Adam(beta1 = nan_value)
match nan_adam
    dnn.Adam
        adam_nan_rejected = false
    error
        adam_nan_rejected = true
print(adam_nan_rejected)

bool batchnorm_infinity_rejected = false
dnn.BatchNorm | error infinite_batchnorm = dnn.BatchNorm(
    features = 2,
    epsilon = infinity
)
match infinite_batchnorm
    dnn.BatchNorm
        batchnorm_infinity_rejected = false
    error
        batchnorm_infinity_rejected = true
print(batchnorm_infinity_rejected)

bool initializer_count_rejected = false
tensor<float32> | error bad_count = dnn.uniform_weights(
    count = -1, bound = 1.0, seed = 1
)
match bad_count
    tensor<float32>
        initializer_count_rejected = false
    error
        initializer_count_rejected = true
print(initializer_count_rejected)

bool initializer_bound_rejected = false
tensor<float32> | error bad_bound = dnn.uniform_weights(
    count = 1, bound = infinity, seed = 1
)
match bad_bound
    tensor<float32>
        initializer_bound_rejected = false
    error
        initializer_bound_rejected = true
print(initializer_bound_rejected)
QUI

contracts_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/contracts.qui")"
contracts_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$contracts_output" != "$contracts_expected" ]]; then
    echo "unexpected dnn contract output: $contracts_output" >&2
    exit 1
fi

cat > "$TMP/private-bridge.qui" <<'QUI'
import dnn
dnn.dnn_runtime_fast()
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/private-bridge.qui" --json \
    >"$TMP/private-bridge.out" 2>&1
bridge_status=$?
set -e
if [[ "$bridge_status" -ne 1 ]]; then
    echo "DNN runtime bridge leaked through the public package API" >&2
    exit 1
fi
grep -Eq 'UNKNOWN_MODULE_MEMBER|UNKNOWN_NAME' "$TMP/private-bridge.out"

cat > "$TMP/removed-mode-wrapper.qui" <<'QUI'
import dnn
dnn.mode(dnn.mode.fast)
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/removed-mode-wrapper.qui" --json \
    >"$TMP/removed-mode-wrapper.out" 2>&1
mode_status=$?
set -e
if [[ "$mode_status" -ne 1 ]]; then
    echo "removed dnn.mode wrapper is still public" >&2
    exit 1
fi
grep -Eq 'UNKNOWN_MODULE_MEMBER|UNKNOWN_NAME' "$TMP/removed-mode-wrapper.out"

cat > "$TMP/all-reduce-integer.qui" <<'QUI'
import dnn

tensor<int>[] values = [tensor.zeros<int>([1])]
dnn.all_reduce_sum(&values)
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/all-reduce-integer.qui" --json \
    >"$TMP/all-reduce-integer.out" 2>&1
all_reduce_integer_status=$?
set -e
if [[ "$all_reduce_integer_status" -ne 1 ]]; then
    echo "integer all_reduce_sum unexpectedly passed the floating constraint" >&2
    exit 1
fi
grep -Eqi 'floating|constraint|type.*argument' "$TMP/all-reduce-integer.out"

cat > "$TMP/logits.qui" <<'QUI'
import dnn

tensor<float32> raw = tensor.zeros<float32>([1, 2])
raw[0, 0] = float32(-1000.0)
raw[0, 1] = float32(1000.0)
tensor<float32> target = tensor.zeros<float32>([1, 2])
target[0, 0] = float32(0.0)
target[0, 1] = float32(1.0)
neural<float32> loss = dnn.binary_cross_entropy_with_logits(
    neural.track(raw), target
)
float32 value = loss.untrack().item()
print(value == value)
print(value < float32(0.001))
QUI

logits_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/logits.qui")"
logits_expected="$(printf 'true\ntrue')"
if [[ "$logits_output" != "$logits_expected" ]]; then
    echo "unexpected dnn logits output: $logits_output" >&2
    exit 1
fi

echo "dnn contract tests: ok"
