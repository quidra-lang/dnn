#!/usr/bin/env bash
set -euo pipefail
QUIDRA="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"

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
QUI
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/reference-models.qui"

cat > "$TMP/root-leak.qui" <<'QUI'
import dnn
dnn.ResNet50 model = dnn.ResNet50()
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/root-leak.qui" >"$TMP/root-leak.out" 2>&1
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
    echo "DNN model namespace leaked into package root" >&2
    exit 1
fi

# A DNN architecture guard returns a structured package error with its code.
cat > "$TMP/architecture-error.qui" <<'QUI'
import dnn

dnn.model.ResNetBasicIdentityStage | error stage = dnn.model.ResNetBasicIdentityStage(channels = 4, count = 0)
match stage
    dnn.model.ResNetBasicIdentityStage
        print(false)
    error problem
        print(problem.code == "DNN_ARGUMENT")
print(NL)
QUI
code_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/architecture-error.qui")"
if [[ "$code_output" != "true" ]]; then
    printf 'DNN architecture error code mismatch: %s\n' "$code_output" >&2
    exit 1
fi

# The public option-A block reaches its own channel-shape guard after
# two small, valid convolutions. It must fail with DNN's production code.
cat > "$TMP/option-a-shape-error.qui" <<'QUI'
import dnn

int | error exercise_shape_guard()
    dnn.model.ResNetBasicOptionABlock block = try dnn.model.ResNetBasicOptionABlock(
        channels_in = 2, channels_out = 1, stride = 1, seed = 1
    )
    tensor<real32> input = tensor.zeros<real32>([1, 2, 1, 1])
    tensor<real32> output = .block.infer(input)
    return 0

auto result = exercise_shape_guard()
print(result)
print(NL)
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/option-a-shape-error.qui" >"$TMP/option-a-shape-error.out" 2>"$TMP/option-a-shape-error.err"
shape_status=$?
set -e
if [[ "$shape_status" -ne 101 ]]; then
    printf 'DNN shape failure status mismatch: %s\n' "$shape_status" >&2
    cat "$TMP/option-a-shape-error.out" "$TMP/option-a-shape-error.err" >&2
    exit 1
fi
grep -F 'Quidra runtime error[DNN_SHAPE]' "$TMP/option-a-shape-error.err"
grep -F 'ResNet option-A shortcut cannot reduce channel count' "$TMP/option-a-shape-error.err"

echo "dnn contracts: ok"
