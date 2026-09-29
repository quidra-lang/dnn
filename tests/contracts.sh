#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 - "$REPOSITORY_ROOT" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
failures = []

def validate_table(path, table):
    if not table:
        return
    expected = len(re.findall(r"(?<!\\)\|", table[0][1]))
    for line_number, line in table:
        actual = len(re.findall(r"(?<!\\)\|", line))
        if actual != expected:
            failures.append(
                f"{path.relative_to(root)}:{line_number}: "
                f"Markdown table has {actual} unescaped separators; expected {expected}"
            )

for path in sorted(root.rglob("*.md")):
    source = path.read_text(encoding="utf-8").splitlines()
    in_fence = False
    table = []
    for line_number, line in enumerate(source, 1):
        if line.lstrip().startswith("```"):
            validate_table(path, table)
            table = []
            in_fence = not in_fence
            continue
        if not in_fence and line.lstrip().startswith("|"):
            table.append((line_number, line))
        else:
            validate_table(path, table)
            table = []
    validate_table(path, table)

if failures:
    print("\n".join(failures), file=sys.stderr)
    raise SystemExit(1)
PY
python3 "$REPOSITORY_ROOT/tests/reference_architectures.py" "$REPOSITORY_ROOT/main.qui"

cat > "$TMP/reference-models.qui" <<'QUI'
import dnn

dnn.model.AlexNet alexnet = dnn.model.AlexNet()
dnn.model.VGGA vgg_a = dnn.model.VGGA()
dnn.model.VGGALRN vgg_a_lrn = dnn.model.VGGALRN()
dnn.model.VGGB vgg_b = dnn.model.VGGB()
dnn.model.VGGC vgg_c = dnn.model.VGGC()
dnn.model.VGGD vgg_d = dnn.model.VGGD()
dnn.model.VGGE vgg_e = dnn.model.VGGE()
dnn.model.VGG11 vgg11 = dnn.model.VGG11()
dnn.model.VGG13 vgg13 = dnn.model.VGG13()
dnn.model.VGG16 vgg16 = dnn.model.VGG16()
dnn.model.VGG19 vgg19 = dnn.model.VGG19()
dnn.model.ResNet18 resnet18 = dnn.model.ResNet18()
dnn.model.ResNet34 resnet34 = dnn.model.ResNet34()
dnn.model.ResNet50 resnet50 = dnn.model.ResNet50()
dnn.model.ResNet101 resnet101 = dnn.model.ResNet101()
dnn.model.ResNet152 resnet152 = dnn.model.ResNet152()

vgg_b.initialize_from(vgg_a)
vgg_c.initialize_from(vgg_a)
vgg_d.initialize_from(vgg_a)
vgg_e.initialize_from(vgg_a)
vgg13.initialize_from(vgg11)
vgg16.initialize_from(vgg11)
vgg19.initialize_from(vgg11)
QUI
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/reference-models.qui"

cat > "$TMP/reference-model-root-leak.qui" <<'QUI'
import dnn

dnn.ResNet50 model = dnn.ResNet50()
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/reference-model-root-leak.qui" --json \
    >"$TMP/reference-model-root-leak.out" 2>&1
root_model_status=$?
set -e
if [[ "$root_model_status" -ne 1 ]]; then
    echo "reference model unexpectedly remained in the dnn root namespace" >&2
    exit 1
fi
grep -Eq 'UNKNOWN_MODULE_MEMBER|UNKNOWN_TYPE|UNKNOWN_NAME' "$TMP/reference-model-root-leak.out"

cat > "$TMP/reference-helper-root-leak.qui" <<'QUI'
import dnn

dnn.ResNetStem helper = dnn.ResNetStem()
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/reference-helper-root-leak.qui" --json \
    >"$TMP/reference-helper-root-leak.out" 2>&1
root_helper_status=$?
set -e
if [[ "$root_helper_status" -ne 1 ]]; then
    echo "reference helper unexpectedly remained in the dnn root namespace" >&2
    exit 1
fi
grep -Eq 'UNKNOWN_MODULE_MEMBER|UNKNOWN_TYPE|UNKNOWN_NAME' "$TMP/reference-helper-root-leak.out"

cat > "$TMP/reference-function-root-leak.qui" <<'QUI'
import dnn

tensor<float32> pixels = tensor.ones<float32>([1, 3, 224, 224])
dnn.require_imagenet_input(pixels, 224)
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/reference-function-root-leak.qui" --json \
    >"$TMP/reference-function-root-leak.out" 2>&1
reference_function_status=$?
set -e
if [[ "$reference_function_status" -ne 1 ]]; then
    echo "reference function helper unexpectedly remained in the dnn root namespace" >&2
    exit 1
fi
grep -Eq 'UNKNOWN_MODULE_MEMBER|UNKNOWN_NAME' "$TMP/reference-function-root-leak.out"

cat > "$TMP/persistence-helper-root-leak.qui" <<'QUI'
import dnn

void | error probe()
    file.Handle input = try file.open("unused")
    tensor<float32> value = try dnn.read_state_tensor32(&input)
    return
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/persistence-helper-root-leak.qui" --json \
    >"$TMP/persistence-helper-root-leak.out" 2>&1
persistence_helper_status=$?
set -e
if [[ "$persistence_helper_status" -ne 1 ]]; then
    echo "persistence helper unexpectedly remained in the dnn root namespace" >&2
    exit 1
fi
grep -Eq 'UNKNOWN_MODULE_MEMBER|UNKNOWN_NAME' "$TMP/persistence-helper-root-leak.out"

cat > "$TMP/contracts.qui" <<'QUI'
import dnn

dnn.mode.fast()
dnn.mode.deterministic()
dnn.mode.fast()

bool linear_rejected = false
dnn.FC | error bad_linear = dnn.FC(features_in = 2, features_out = 0)
match bad_linear
    dnn.FC
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
dnn.FC | error huge_linear = dnn.FC(
    features_in = 3037000500,
    features_out = 3037000500
)
match huge_linear
    dnn.FC
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

bool normal_initializer_scale_rejected = false
tensor<float32> | error bad_normal_scale = dnn.normal_weights(
    count = 1, standard_deviation = infinity, seed = 1
)
match bad_normal_scale
    tensor<float32>
        normal_initializer_scale_rejected = false
    error
        normal_initializer_scale_rejected = true
print(normal_initializer_scale_rejected)

bool normal_initializer_finite = false
tensor<float32> | error normal_sample = dnn.normal_weights(
    count = 2, standard_deviation = 1.0, seed = 17
)
match normal_sample
    tensor<float32>
        normal_initializer_finite = (
            math.is_finite(float(normal_sample[0].item()))
            and math.is_finite(float(normal_sample[1].item()))
        )
    error
        normal_initializer_finite = false
print(normal_initializer_finite)

dnn.FC normal_dense = dnn.normal_fc(
    features_in = 2,
    features_out = 1,
    standard_deviation = 0.0,
    bias_value = 1.0,
    seed = 5
)
print(normal_dense.weight.raw()[0, 0].item() == float32(0))
print(normal_dense.bias.raw()[0].item() == float32(1))

dnn.Conv2D normal_convolution = dnn.normal_conv2d(
    channels_in = 1,
    channels_out = 1,
    kernel = 1,
    standard_deviation = 0.0,
    bias_value = 1.0,
    seed = 5
)
print(normal_convolution.weight.raw()[0, 0, 0, 0].item() == float32(0))
print(normal_convolution.bias.raw()[0].item() == float32(1))

dnn.model.ResNetBasicIdentityStage compact_stage = dnn.model.ResNetBasicIdentityStage(
    channels = 2,
    count = 2,
    seed = 11
)
dnn.Parameter<float32>[] compact_parameters = reflect.collect<dnn.Parameter<float32>>(
    compact_stage
)
dnn.State<float32>[] compact_states = reflect.collect<dnn.State<float32>>(
    compact_stage
)
print(len(compact_parameters) == 16)
print(len(compact_states) == 8)
QUI

contracts_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/contracts.qui")"
contracts_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$contracts_output" != "$contracts_expected" ]]; then
    echo "unexpected dnn contract output: $contracts_output" >&2
    exit 1
fi

cat > "$TMP/const-sgd-step.qui" <<'QUI'
import dnn

class StepModel
    dnn.Parameter<float32> weight

StepModel model
model.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1]))
const dnn.SGD optimizer = dnn.SGD()
optimizer.step(&model)
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/const-sgd-step.qui" --json \
    >"$TMP/const-sgd-step.out" 2>&1
const_sgd_status=$?
set -e
if [[ "$const_sgd_status" -ne 1 ]]; then
    echo "const SGD unexpectedly allowed mutable step()" >&2
    exit 1
fi
grep -Fq 'WRITE_CAPABILITY' "$TMP/const-sgd-step.out"

cat > "$TMP/const-adam-step.qui" <<'QUI'
import dnn

class StepModel
    dnn.Parameter<float32> weight

StepModel model
model.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1]))
const dnn.Adam optimizer = dnn.Adam()
optimizer.step(&model)
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/const-adam-step.qui" --json \
    >"$TMP/const-adam-step.out" 2>&1
const_adam_status=$?
set -e
if [[ "$const_adam_status" -ne 1 ]]; then
    echo "const Adam unexpectedly allowed mutable step()" >&2
    exit 1
fi
grep -Fq 'WRITE_CAPABILITY' "$TMP/const-adam-step.out"

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

cat > "$TMP/logits.qui" <<'QUI'
import dnn

tensor<float32> raw = tensor.zeros<float32>([1, 2])
raw[0, 0] = float32(-1000.0)
raw[0, 1] = float32(1000.0)
tensor<float32> target = tensor.zeros<float32>([1, 2])
target[0, 0] = float32(0.0)
target[0, 1] = float32(1.0)
tensor<float32> loss = dnn.binary_cross_entropy_with_logits(
    raw.track(), target
)
float32 value = loss.item()
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
