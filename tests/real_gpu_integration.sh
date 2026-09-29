#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:-}"
if [[ -z "$QUIDRA" ]]; then
    echo "usage: $0 /path/to/quidra" >&2
    exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
GPU_INDEX="${QUIDRA_REAL_GPU_INDEX:-0}"
REQUIRE_REAL="${QUIDRA_REQUIRE_REAL_GPU:-0}"
REQUIRE_BACKEND="${QUIDRA_REQUIRE_GPU_BACKEND:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

set +e
gpu_info="$("$QUIDRA" gpu 2>&1)"
gpu_status=$?
set -e
skip_or_fail() {
    local reason="$1"
    if [[ "$REQUIRE_REAL" == "1" ]]; then
        echo "real DNN GPU integration required but unavailable: $reason" >&2
        printf '%s\n' "$gpu_info" >&2
        exit 1
    fi
    echo "dnn real GPU integration: skipped ($reason)"
    exit 0
}
if [[ $gpu_status -ne 0 ]]; then skip_or_fail "quidra gpu failed"; fi
if grep -Fq "backend: TEST" <<<"$gpu_info"; then skip_or_fail "fake GPU backend is active"; fi
if ! grep -Fq "GPU $GPU_INDEX" <<<"$gpu_info"; then skip_or_fail "gpu($GPU_INDEX) is not present"; fi
gpu_block="$(awk -v target="GPU $GPU_INDEX" '
    $0 == target { found = 1; print; next }
    found && /^GPU [0-9]+$/ { exit }
    found { print }
' <<<"$gpu_info")"
if [[ -n "$REQUIRE_BACKEND" ]] && ! grep -Fq "backend: $REQUIRE_BACKEND" <<<"$gpu_block"; then
    skip_or_fail "gpu($GPU_INDEX) is not backend $REQUIRE_BACKEND"
fi

cat > "$TMP/dnn-real-gpu.qui" <<QUI
import dnn

class FCModel
    dnn.FC dense

class ConvModel
    dnn.Conv2D convolution

class NormModel
    dnn.BatchNorm normalization

bool near(float32 a, float32 b)
    float32 d = a - b
    return d > float32(-0.0005) and d < float32(0.0005)

int | error run()
    dnn.FC cpu_linear
    cpu_linear.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1, 2]))
    cpu_linear.bias = dnn.Parameter<float32>(value = tensor.zeros<float32>([1]))
    dnn.FC gpu_linear
    gpu_linear.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1, 2], gpu = $GPU_INDEX))
    gpu_linear.bias = dnn.Parameter<float32>(value = tensor.zeros<float32>([1], gpu = $GPU_INDEX))
    tensor<float32> cpu_samples = tensor.ones<float32>([1, 2])
    tensor<float32> gpu_samples = cpu_samples.gpu($GPU_INDEX)
    tensor<float32> cpu_linear_out = cpu_linear.forward(cpu_samples)
    tensor<float32> gpu_linear_out = gpu_linear.forward(gpu_samples).cpu()
    print(near(cpu_linear_out[0, 0].item(), gpu_linear_out[0, 0].item()))

    dnn.Conv2D cpu_conv
    cpu_conv.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1, 1, 1, 1]))
    cpu_conv.bias = dnn.Parameter<float32>(value = tensor.zeros<float32>([1]))
    cpu_conv.step = 1
    cpu_conv.border = 0
    dnn.Conv2D gpu_conv
    gpu_conv.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1, 1, 1, 1], gpu = $GPU_INDEX))
    gpu_conv.bias = dnn.Parameter<float32>(value = tensor.zeros<float32>([1], gpu = $GPU_INDEX))
    gpu_conv.step = 1
    gpu_conv.border = 0
    tensor<float32> cpu_pixels = tensor.ones<float32>([1, 1, 2, 2])
    tensor<float32> gpu_pixels = cpu_pixels.gpu($GPU_INDEX)
    tensor<float32> cpu_conv_out = cpu_conv.forward(cpu_pixels)
    tensor<float32> gpu_conv_out = gpu_conv.forward(gpu_pixels).cpu()
    print(near(cpu_conv_out[0, 0, 1, 1].item(), gpu_conv_out[0, 0, 1, 1].item()))

    dnn.Conv2D cpu_conv3
    cpu_conv3.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([2, 2, 3, 3]))
    cpu_conv3.bias = dnn.Parameter<float32>(value = tensor.zeros<float32>([2]))
    cpu_conv3.step = 2
    cpu_conv3.border = 1
    dnn.Conv2D gpu_conv3
    gpu_conv3.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([2, 2, 3, 3], gpu = $GPU_INDEX))
    gpu_conv3.bias = dnn.Parameter<float32>(value = tensor.zeros<float32>([2], gpu = $GPU_INDEX))
    gpu_conv3.step = 2
    gpu_conv3.border = 1
    tensor<float32> cpu_conv3_input = tensor.ones<float32>([2, 2, 5, 7])
    tensor<float32> gpu_conv3_input = cpu_conv3_input.gpu($GPU_INDEX)
    tensor<float32> cpu_conv3_out = cpu_conv3.forward(cpu_conv3_input)
    tensor<float32> gpu_conv3_out = gpu_conv3.forward(gpu_conv3_input).cpu()
    print(
        cpu_conv3_out.shape()[2] == gpu_conv3_out.shape()[2] and
        cpu_conv3_out.shape()[3] == gpu_conv3_out.shape()[3] and
        near(cpu_conv3_out[0, 0, 0, 0].item(), gpu_conv3_out[0, 0, 0, 0].item()) and
        near(cpu_conv3_out[1, 1, 1, 1].item(), gpu_conv3_out[1, 1, 1, 1].item())
    )
    dnn.mode.deterministic()
    tensor<float32> deterministic_first = gpu_conv3.forward(gpu_conv3_input).cpu()
    tensor<float32> deterministic_second = gpu_conv3.forward(gpu_conv3_input).cpu()
    print(
        deterministic_first[0, 0, 0, 0].item() ==
        deterministic_second[0, 0, 0, 0].item() and
        deterministic_first[1, 1, 1, 1].item() ==
        deterministic_second[1, 1, 1, 1].item()
    )
    dnn.mode.fast()

    dnn.BatchNorm cpu_norm
    cpu_norm.scale = dnn.Parameter<float32>(value = tensor.ones<float32>([2]))
    cpu_norm.bias = dnn.Parameter<float32>(value = tensor.zeros<float32>([2]))
    cpu_norm.running_mean = dnn.State<float32>(value = tensor.zeros<float32>([2]))
    cpu_norm.running_variance = dnn.State<float32>(value = tensor.ones<float32>([2]))
    cpu_norm.running_momentum = 0.1
    cpu_norm.variance_epsilon = 0.00001
    dnn.BatchNorm gpu_norm
    gpu_norm.scale = dnn.Parameter<float32>(value = tensor.ones<float32>([2], gpu = $GPU_INDEX))
    gpu_norm.bias = dnn.Parameter<float32>(value = tensor.zeros<float32>([2], gpu = $GPU_INDEX))
    gpu_norm.running_mean = dnn.State<float32>(value = tensor.zeros<float32>([2], gpu = $GPU_INDEX))
    gpu_norm.running_variance = dnn.State<float32>(value = tensor.ones<float32>([2], gpu = $GPU_INDEX))
    gpu_norm.running_momentum = 0.1
    gpu_norm.variance_epsilon = 0.00001
    tensor<float32> cpu_norm_input = tensor.ones<float32>([2, 2])
    tensor<float32> gpu_norm_input = cpu_norm_input.gpu($GPU_INDEX)
    tensor<float32> cpu_norm_out = cpu_norm.infer(cpu_norm_input)
    tensor<float32> gpu_norm_out = gpu_norm.infer(gpu_norm_input).cpu()
    print(near(cpu_norm_out[0, 0].item(), gpu_norm_out[0, 0].item()))

    dnn.Dropout cpu_drop = dnn.Dropout(rate = 0.5, seed = 17)
    dnn.Dropout gpu_drop = dnn.Dropout(rate = 0.5, seed = 17)
    tensor<float32> drop_input = tensor.ones<float32>([2, 2])
    tensor<float32> cpu_dropped = cpu_drop.forward(drop_input.track()).untrack()
    tensor<float32> gpu_dropped = gpu_drop.forward(drop_input.gpu($GPU_INDEX).track()).untrack().cpu()
    print(near(cpu_dropped[0, 0].item(), gpu_dropped[0, 0].item()))
    print(near(cpu_dropped[1, 1].item(), gpu_dropped[1, 1].item()))
    tensor<float32> cpu_dropped_next = cpu_drop.forward(drop_input.track()).untrack()
    tensor<float32> gpu_dropped_next = gpu_drop.forward(drop_input.gpu($GPU_INDEX).track()).untrack().cpu()
    print(near(cpu_dropped_next[0, 1].item(), gpu_dropped_next[0, 1].item()))

    tensor<float32> cpu_activation = tensor.zeros<float32>([1, 2])
    cpu_activation[0, 0] = float32(-1)
    cpu_activation[0, 1] = float32(1)
    tensor<float32> gpu_activation = cpu_activation.gpu($GPU_INDEX)

    tensor<float32> cpu_relu = dnn.relu(cpu_activation.track()).untrack()
    tensor<float32> gpu_relu = dnn.relu(gpu_activation.track()).untrack().cpu()
    print(near(cpu_relu[0, 0].item(), gpu_relu[0, 0].item()) and near(cpu_relu[0, 1].item(), gpu_relu[0, 1].item()))

    tensor<float32> cpu_tanh = dnn.tanh(cpu_activation.track()).untrack()
    tensor<float32> gpu_tanh = dnn.tanh(gpu_activation.track()).untrack().cpu()
    print(near(cpu_tanh[0, 0].item(), gpu_tanh[0, 0].item()) and near(cpu_tanh[0, 1].item(), gpu_tanh[0, 1].item()))

    tensor<float32> cpu_sigmoid = dnn.sigmoid(cpu_activation.track()).untrack()
    tensor<float32> gpu_sigmoid = dnn.sigmoid(gpu_activation.track()).untrack().cpu()
    print(near(cpu_sigmoid[0, 0].item(), gpu_sigmoid[0, 0].item()) and near(cpu_sigmoid[0, 1].item(), gpu_sigmoid[0, 1].item()))

    tensor<float32> cpu_softmax = dnn.softmax(cpu_activation.track()).untrack()
    tensor<float32> gpu_softmax = dnn.softmax(gpu_activation.track()).untrack().cpu()
    print(near(cpu_softmax[0, 0].item(), gpu_softmax[0, 0].item()) and near(cpu_softmax[0, 1].item(), gpu_softmax[0, 1].item()))

    tensor<float32> cpu_gelu = dnn.gelu(cpu_activation.track()).untrack()
    tensor<float32> gpu_gelu = dnn.gelu(gpu_activation.track()).untrack().cpu()
    print(near(cpu_gelu[0, 0].item(), gpu_gelu[0, 0].item()) and near(cpu_gelu[0, 1].item(), gpu_gelu[0, 1].item()))

    tensor<float32> cpu_probability = tensor.zeros<float32>([1, 2])
    cpu_probability[0, 0] = float32(0.25)
    cpu_probability[0, 1] = float32(0.75)
    tensor<float32> gpu_probability = cpu_probability.gpu($GPU_INDEX)
    tensor<float32> cpu_binary_target = tensor.zeros<float32>([1, 2])
    cpu_binary_target[0, 1] = float32(1)
    tensor<float32> gpu_binary_target = cpu_binary_target.gpu($GPU_INDEX)

    float32 cpu_mse_loss = dnn.mse(cpu_probability.track(), cpu_binary_target).item()
    float32 gpu_mse_loss = dnn.mse(gpu_probability.track(), gpu_binary_target).item()
    print(near(cpu_mse_loss, gpu_mse_loss))

    float32 cpu_bce = dnn.binary_cross_entropy(cpu_probability.track(), cpu_binary_target).item()
    float32 gpu_bce = dnn.binary_cross_entropy(gpu_probability.track(), gpu_binary_target).item()
    print(near(cpu_bce, gpu_bce))

    float32 cpu_bce_logits = dnn.binary_cross_entropy_with_logits(cpu_activation.track(), cpu_binary_target).item()
    float32 gpu_bce_logits = dnn.binary_cross_entropy_with_logits(gpu_activation.track(), gpu_binary_target).item()
    print(near(cpu_bce_logits, gpu_bce_logits))

    float32 cpu_cross_entropy = dnn.cross_entropy(cpu_activation.track(), cpu_binary_target).item()
    float32 gpu_cross_entropy = dnn.cross_entropy(gpu_activation.track(), gpu_binary_target).item()
    print(near(cpu_cross_entropy, gpu_cross_entropy))

    FCModel cpu_model
    cpu_model.dense = cpu_linear
    FCModel gpu_model
    gpu_model.dense = gpu_linear
    tensor<float32> cpu_target = tensor.zeros<float32>([1, 1])
    tensor<float32> gpu_target = cpu_target.gpu($GPU_INDEX)
    tensor<float32> cpu_loss = dnn.mse(
        cpu_model.dense.forward(cpu_samples.track()), cpu_target
    )
    tensor<float32> gpu_loss = dnn.mse(
        gpu_model.dense.forward(gpu_samples.track()), gpu_target
    )
    dnn.SGD cpu_sgd = try dnn.SGD(rate = 0.1)
    dnn.SGD gpu_sgd = try dnn.SGD(rate = 0.1)
    cpu_sgd.zero_grad(&cpu_model)
    gpu_sgd.zero_grad(&gpu_model)
    cpu_loss.backward(&cpu_model)
    gpu_loss.backward(&gpu_model)
    cpu_sgd.step(&cpu_model)
    gpu_sgd.step(&gpu_model)
    print(near(
        cpu_model.dense.weight.raw()[0, 0].item(),
        gpu_model.dense.weight.raw()[0, 0].item()
    ))

    ConvModel cpu_conv_model
    cpu_conv_model.convolution = cpu_conv
    ConvModel gpu_conv_model
    gpu_conv_model.convolution = gpu_conv
    tensor<float32> cpu_zero_image = tensor.zeros<float32>([1, 1, 2, 2])
    tensor<float32> gpu_zero_image = cpu_zero_image.gpu($GPU_INDEX)
    tensor<float32> cpu_conv_loss = dnn.mse(
        cpu_conv_model.convolution.forward(cpu_pixels.track()), cpu_zero_image
    )
    tensor<float32> gpu_conv_loss = dnn.mse(
        gpu_conv_model.convolution.forward(gpu_pixels.track()), gpu_zero_image
    )
    dnn.SGD cpu_conv_sgd = try dnn.SGD(rate = 0.1)
    dnn.SGD gpu_conv_sgd = try dnn.SGD(rate = 0.1)
    cpu_conv_sgd.zero_grad(&cpu_conv_model)
    gpu_conv_sgd.zero_grad(&gpu_conv_model)
    cpu_conv_loss.backward(&cpu_conv_model)
    gpu_conv_loss.backward(&gpu_conv_model)
    cpu_conv_sgd.step(&cpu_conv_model)
    gpu_conv_sgd.step(&gpu_conv_model)
    print(near(
        cpu_conv_model.convolution.weight.raw()[0, 0, 0, 0].item(),
        gpu_conv_model.convolution.weight.raw()[0, 0, 0, 0].item()
    ))

    ConvModel cpu_conv3_model
    cpu_conv3_model.convolution = cpu_conv3
    ConvModel gpu_conv3_model
    gpu_conv3_model.convolution = gpu_conv3
    tensor<float32> cpu_conv3_target = tensor.zeros<float32>([2, 2, 3, 4])
    tensor<float32> gpu_conv3_target = cpu_conv3_target.gpu($GPU_INDEX)
    tensor<float32> cpu_conv3_loss = dnn.mse(
        cpu_conv3_model.convolution.forward(cpu_conv3_input.track()),
        cpu_conv3_target
    )
    tensor<float32> gpu_conv3_loss = dnn.mse(
        gpu_conv3_model.convolution.forward(gpu_conv3_input.track()),
        gpu_conv3_target
    )
    dnn.SGD cpu_conv3_sgd = try dnn.SGD(rate = 0.01)
    dnn.SGD gpu_conv3_sgd = try dnn.SGD(rate = 0.01)
    cpu_conv3_sgd.zero_grad(&cpu_conv3_model)
    gpu_conv3_sgd.zero_grad(&gpu_conv3_model)
    cpu_conv3_loss.backward(&cpu_conv3_model)
    gpu_conv3_loss.backward(&gpu_conv3_model)
    cpu_conv3_sgd.step(&cpu_conv3_model)
    gpu_conv3_sgd.step(&gpu_conv3_model)
    print(near(
        cpu_conv3_model.convolution.weight.raw()[0, 0, 1, 1].item(),
        gpu_conv3_model.convolution.weight.raw()[0, 0, 1, 1].item()
    ))
    print(near(
        cpu_conv3_model.convolution.bias.raw()[1].item(),
        gpu_conv3_model.convolution.bias.raw()[1].item()
    ))

    NormModel cpu_norm_model
    cpu_norm_model.normalization = cpu_norm
    NormModel gpu_norm_model
    gpu_norm_model.normalization = gpu_norm
    tensor<float32> cpu_norm_train = cpu_norm_model.normalization.forward(cpu_norm_input.track())
    tensor<float32> gpu_norm_train = gpu_norm_model.normalization.forward(gpu_norm_input.track())
    tensor<float32> cpu_norm_loss = (cpu_norm_train * cpu_norm_train).mean()
    tensor<float32> gpu_norm_loss = (gpu_norm_train * gpu_norm_train).mean()
    dnn.SGD cpu_norm_sgd = try dnn.SGD(rate = 0.1)
    dnn.SGD gpu_norm_sgd = try dnn.SGD(rate = 0.1)
    cpu_norm_sgd.zero_grad(&cpu_norm_model)
    gpu_norm_sgd.zero_grad(&gpu_norm_model)
    cpu_norm_loss.backward(&cpu_norm_model)
    gpu_norm_loss.backward(&gpu_norm_model)
    cpu_norm_sgd.step(&cpu_norm_model)
    gpu_norm_sgd.step(&gpu_norm_model)
    print(near(
        cpu_norm_model.normalization.scale.raw()[0].item(),
        gpu_norm_model.normalization.scale.raw()[0].item()
    ))
    tensor<float32> cpu_norm_infer = cpu_norm_model.normalization.infer(cpu_norm_input)
    tensor<float32> gpu_norm_infer = gpu_norm_model.normalization.infer(gpu_norm_input).cpu()
    print(near(cpu_norm_infer[0, 0].item(), gpu_norm_infer[0, 0].item()))

    dnn.FC cpu_adam_layer
    cpu_adam_layer.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1, 1]))
    cpu_adam_layer.bias = dnn.Parameter<float32>(value = tensor.zeros<float32>([1]))
    dnn.FC gpu_adam_layer
    gpu_adam_layer.weight = dnn.Parameter<float32>(value = tensor.ones<float32>([1, 1], gpu = $GPU_INDEX))
    gpu_adam_layer.bias = dnn.Parameter<float32>(value = tensor.zeros<float32>([1], gpu = $GPU_INDEX))
    FCModel cpu_adam_model
    cpu_adam_model.dense = cpu_adam_layer
    FCModel gpu_adam_model
    gpu_adam_model.dense = gpu_adam_layer
    tensor<float32> cpu_one = tensor.ones<float32>([1, 1])
    tensor<float32> gpu_one = cpu_one.gpu($GPU_INDEX)
    tensor<float32> cpu_zero = tensor.zeros<float32>([1, 1])
    tensor<float32> gpu_zero = cpu_zero.gpu($GPU_INDEX)
    tensor<float32> cpu_adam_loss = dnn.mse(
        cpu_adam_model.dense.forward(cpu_one.track()), cpu_zero
    )
    tensor<float32> gpu_adam_loss = dnn.mse(
        gpu_adam_model.dense.forward(gpu_one.track()), gpu_zero
    )
    dnn.Adam cpu_adam = try dnn.Adam(rate = 0.1)
    dnn.Adam gpu_adam = try dnn.Adam(rate = 0.1)
    cpu_adam.zero_grad(&cpu_adam_model)
    gpu_adam.zero_grad(&gpu_adam_model)
    cpu_adam_loss.backward(&cpu_adam_model)
    gpu_adam_loss.backward(&gpu_adam_model)
    cpu_adam.step(&cpu_adam_model)
    gpu_adam.step(&gpu_adam_model)
    print(near(
        cpu_adam_model.dense.weight.raw()[0, 0].item(),
        gpu_adam_model.dense.weight.raw()[0, 0].item()
    ))
    print(near(
        cpu_adam_model.dense.bias.raw()[0].item(),
        gpu_adam_model.dense.bias.raw()[0].item()
    ))

    tensor<float32> cpu_adam_loss2 = dnn.mse(
        cpu_adam_model.dense.forward(cpu_one.track()), cpu_zero
    )
    tensor<float32> gpu_adam_loss2 = dnn.mse(
        gpu_adam_model.dense.forward(gpu_one.track()), gpu_zero
    )
    cpu_adam.zero_grad(&cpu_adam_model)
    gpu_adam.zero_grad(&gpu_adam_model)
    cpu_adam_loss2.backward(&cpu_adam_model)
    gpu_adam_loss2.backward(&gpu_adam_model)
    cpu_adam.step(&cpu_adam_model)
    gpu_adam.step(&gpu_adam_model)
    print(near(
        cpu_adam_model.dense.weight.raw()[0, 0].item(),
        gpu_adam_model.dense.weight.raw()[0, 0].item()
    ))
    print(near(
        cpu_adam_model.dense.bias.raw()[0].item(),
        gpu_adam_model.dense.bias.raw()[0].item()
    ))
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/dnn-real-gpu.qui")"
expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$output" != "$expected" ]]; then
    echo "DNN real GPU numerical equivalence failed on gpu($GPU_INDEX)" >&2
    printf '%s\n' "$output" >&2
    exit 1
fi

echo "dnn real GPU integration: ok on gpu($GPU_INDEX)"
