#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/contracts.qui" <<'QUI'
import dnn

dnn.mode(dnn.fast)
dnn.mode(dnn.deterministic)
dnn.mode(dnn.fast)

bool linear_rejected = false
dnn.LinearLayer | error bad_linear = dnn.Linear(features_in = 2, features_out = 0)
match bad_linear
    dnn.LinearLayer
        linear_rejected = false
    error
        linear_rejected = true
print(linear_rejected)

bool convolution_rejected = false
dnn.Conv2DLayer | error bad_convolution = dnn.Conv2D(
    channels_in = 1,
    channels_out = 1,
    kernel = 3,
    stride = 0
)
match bad_convolution
    dnn.Conv2DLayer
        convolution_rejected = false
    error
        convolution_rejected = true
print(convolution_rejected)

bool normalization_rejected = false
dnn.BatchNormLayer | error bad_normalization = dnn.BatchNorm(
    features = 2,
    momentum = 1.5
)
match bad_normalization
    dnn.BatchNormLayer
        normalization_rejected = false
    error
        normalization_rejected = true
print(normalization_rejected)

bool dropout_rejected = false
dnn.DropoutLayer | error bad_dropout = dnn.Dropout(rate = 1.0)
match bad_dropout
    dnn.DropoutLayer
        dropout_rejected = false
    error
        dropout_rejected = true
print(dropout_rejected)

bool sgd_rejected = false
dnn.SGDOptimizer | error bad_sgd = dnn.SGD(rate = 0.0)
match bad_sgd
    dnn.SGDOptimizer
        sgd_rejected = false
    error
        sgd_rejected = true
print(sgd_rejected)

bool adam_rejected = false
dnn.AdamOptimizer | error bad_adam = dnn.Adam(beta1 = 1.0)
match bad_adam
    dnn.AdamOptimizer
        adam_rejected = false
    error
        adam_rejected = true
print(adam_rejected)
QUI

contracts_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/contracts.qui")"
contracts_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue')"
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
