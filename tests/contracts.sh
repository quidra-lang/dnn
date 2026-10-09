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

echo "dnn contracts: ok"
