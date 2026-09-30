#!/usr/bin/env bash
set -euo pipefail
QUIDRA="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/model.qui" <<'QUI'
import dnn
import nn

int | error run()
    dnn.model.ResNetBasicBlock block = try dnn.model.ResNetBasicBlock(
        channels = 2,
        seed = 1
    )
    tensor<float32> input = tensor.ones<float32>([1, 2, 4, 4])
    tensor<float32> output = block.infer(input)
    int[] expected_shape = [1, 2, 4, 4]
    print(output.shape() == expected_shape)
    print(NL)
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
        print(NL)
QUI

out="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/model.qui")"
if [[ "$out" != "true" ]]; then
    echo "DNN model integration failed: $out" >&2
    exit 1
fi

AOT="$TMP/model-aot"
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" build "$TMP/model.qui" -o "$AOT"
aot_out="$("$AOT")"
if [[ "$aot_out" != "true" ]]; then
    echo "DNN model AOT integration failed: $aot_out" >&2
    exit 1
fi

repl_out="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" repl < "$TMP/model.qui"
)"
if ! grep -Fxq "true" <<< "$repl_out"; then
    echo "DNN package REPL/JIT integration failed:" >&2
    printf '%s\n' "$repl_out" >&2
    exit 1
fi

cat > "$TMP/no-generic-root.qui" <<'QUI'
import dnn
dnn.FC layer = dnn.FC(2, 1)
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/no-generic-root.qui" >"$TMP/no-generic-root.out" 2>&1
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
    echo "DNN still exposes generic NN layer API" >&2
    exit 1
fi
if ! grep -Eq "UNKNOWN_TYPE|UNKNOWN_MODULE_MEMBER|UNKNOWN_MEMBER|UNKNOWN_NAME" "$TMP/no-generic-root.out"; then
    echo "DNN generic-API rejection used an unexpected diagnostic" >&2
    cat "$TMP/no-generic-root.out" >&2
    exit 1
fi

echo "dnn integration: ok"
