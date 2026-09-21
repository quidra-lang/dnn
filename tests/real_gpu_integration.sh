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
REQUIRE_MULTI="${QUIDRA_REQUIRE_MULTI_GPU:-0}"
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

class LinearModel
    dnn.LinearLayer dense

class ConvModel
    dnn.Conv2DLayer convolution

class NormModel
    dnn.BatchNormLayer normalization

bool near(float32 a, float32 b)
    float32 d = a - b
    return d > float32(-0.0005) and d < float32(0.0005)

int | error run()
    dnn.LinearLayer cpu_linear = dnn.LinearLayer(
        weight = neural.Parameter<float32>(value = tensor.ones<float32>([1, 2])),
        bias = neural.Parameter<float32>(value = tensor.zeros<float32>([1]))
    )
    dnn.LinearLayer gpu_linear = dnn.LinearLayer(
        weight = neural.Parameter<float32>(value = tensor.ones<float32>([1, 2], gpu = $GPU_INDEX)),
        bias = neural.Parameter<float32>(value = tensor.zeros<float32>([1], gpu = $GPU_INDEX))
    )
    tensor<float32> cpu_samples = tensor.ones<float32>([1, 2])
    tensor<float32> gpu_samples = cpu_samples.gpu($GPU_INDEX)
    tensor<float32> cpu_linear_out = cpu_linear.forward(cpu_samples)
    tensor<float32> gpu_linear_out = gpu_linear.forward(gpu_samples).cpu()
    print(near(cpu_linear_out[0, 0].item(), gpu_linear_out[0, 0].item()))

    dnn.Conv2DLayer cpu_conv = dnn.Conv2DLayer(
        weight = neural.Parameter<float32>(value = tensor.ones<float32>([1, 1, 1, 1])),
        bias = neural.Parameter<float32>(value = tensor.zeros<float32>([1])),
        stride = 1,
        padding = 0
    )
    dnn.Conv2DLayer gpu_conv = dnn.Conv2DLayer(
        weight = neural.Parameter<float32>(value = tensor.ones<float32>([1, 1, 1, 1], gpu = $GPU_INDEX)),
        bias = neural.Parameter<float32>(value = tensor.zeros<float32>([1], gpu = $GPU_INDEX)),
        stride = 1,
        padding = 0
    )
    tensor<float32> cpu_pixels = tensor.ones<float32>([1, 1, 2, 2])
    tensor<float32> gpu_pixels = cpu_pixels.gpu($GPU_INDEX)
    tensor<float32> cpu_conv_out = cpu_conv.forward(cpu_pixels)
    tensor<float32> gpu_conv_out = gpu_conv.forward(gpu_pixels).cpu()
    print(near(cpu_conv_out[0, 0, 1, 1].item(), gpu_conv_out[0, 0, 1, 1].item()))

    dnn.Conv2DLayer cpu_conv3 = dnn.Conv2DLayer(
        weight = neural.Parameter<float32>(value = tensor.ones<float32>([2, 2, 3, 3])),
        bias = neural.Parameter<float32>(value = tensor.zeros<float32>([2])),
        stride = 2,
        padding = 1
    )
    dnn.Conv2DLayer gpu_conv3 = dnn.Conv2DLayer(
        weight = neural.Parameter<float32>(value = tensor.ones<float32>([2, 2, 3, 3], gpu = $GPU_INDEX)),
        bias = neural.Parameter<float32>(value = tensor.zeros<float32>([2], gpu = $GPU_INDEX)),
        stride = 2,
        padding = 1
    )
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
    dnn.deterministic()
    tensor<float32> deterministic_first = gpu_conv3.forward(gpu_conv3_input).cpu()
    tensor<float32> deterministic_second = gpu_conv3.forward(gpu_conv3_input).cpu()
    print(
        deterministic_first[0, 0, 0, 0].item() ==
        deterministic_second[0, 0, 0, 0].item() and
        deterministic_first[1, 1, 1, 1].item() ==
        deterministic_second[1, 1, 1, 1].item()
    )
    dnn.fast()

    dnn.BatchNormLayer cpu_norm = dnn.BatchNormLayer(
        scale = neural.Parameter<float32>(value = tensor.ones<float32>([2])),
        bias = neural.Parameter<float32>(value = tensor.zeros<float32>([2])),
        running_mean = neural.State<tensor<float32>>(value = tensor.zeros<float32>([2])),
        running_variance = neural.State<tensor<float32>>(value = tensor.ones<float32>([2])),
        momentum = 0.1,
        epsilon = 0.00001
    )
    dnn.BatchNormLayer gpu_norm = dnn.BatchNormLayer(
        scale = neural.Parameter<float32>(value = tensor.ones<float32>([2], gpu = $GPU_INDEX)),
        bias = neural.Parameter<float32>(value = tensor.zeros<float32>([2], gpu = $GPU_INDEX)),
        running_mean = neural.State<tensor<float32>>(value = tensor.zeros<float32>([2], gpu = $GPU_INDEX)),
        running_variance = neural.State<tensor<float32>>(value = tensor.ones<float32>([2], gpu = $GPU_INDEX)),
        momentum = 0.1,
        epsilon = 0.00001
    )
    tensor<float32> cpu_norm_input = tensor.ones<float32>([2, 2])
    tensor<float32> gpu_norm_input = cpu_norm_input.gpu($GPU_INDEX)
    tensor<float32> cpu_norm_out = cpu_norm.infer(cpu_norm_input)
    tensor<float32> gpu_norm_out = gpu_norm.infer(gpu_norm_input).cpu()
    print(near(cpu_norm_out[0, 0].item(), gpu_norm_out[0, 0].item()))

    dnn.DropoutLayer cpu_drop = dnn.DropoutLayer(
        rate = 0.5, rng = neural.State<uint64>(value = uint64(17))
    )
    dnn.DropoutLayer gpu_drop = dnn.DropoutLayer(
        rate = 0.5, rng = neural.State<uint64>(value = uint64(17))
    )
    tensor<float32> drop_input = tensor.ones<float32>([2, 2])
    tensor<float32> cpu_dropped = cpu_drop.forward(neural.track(drop_input)).untrack()
    tensor<float32> gpu_dropped = gpu_drop.forward(neural.track(drop_input.gpu($GPU_INDEX))).untrack().cpu()
    print(near(cpu_dropped[0, 0].item(), gpu_dropped[0, 0].item()))
    print(near(cpu_dropped[1, 1].item(), gpu_dropped[1, 1].item()))
    tensor<float32> cpu_dropped_next = cpu_drop.forward(neural.track(drop_input)).untrack()
    tensor<float32> gpu_dropped_next = gpu_drop.forward(neural.track(drop_input.gpu($GPU_INDEX))).untrack().cpu()
    print(near(cpu_dropped_next[0, 1].item(), gpu_dropped_next[0, 1].item()))

    tensor<float32> cpu_activation = tensor.zeros<float32>([1, 2])
    cpu_activation[0, 0] = float32(-1)
    cpu_activation[0, 1] = float32(1)
    tensor<float32> gpu_activation = cpu_activation.gpu($GPU_INDEX)

    tensor<float32> cpu_relu = dnn.relu(neural.track(cpu_activation)).untrack()
    tensor<float32> gpu_relu = dnn.relu(neural.track(gpu_activation)).untrack().cpu()
    print(near(cpu_relu[0, 0].item(), gpu_relu[0, 0].item()) and near(cpu_relu[0, 1].item(), gpu_relu[0, 1].item()))

    tensor<float32> cpu_tanh = dnn.tanh(neural.track(cpu_activation)).untrack()
    tensor<float32> gpu_tanh = dnn.tanh(neural.track(gpu_activation)).untrack().cpu()
    print(near(cpu_tanh[0, 0].item(), gpu_tanh[0, 0].item()) and near(cpu_tanh[0, 1].item(), gpu_tanh[0, 1].item()))

    tensor<float32> cpu_sigmoid = dnn.sigmoid(neural.track(cpu_activation)).untrack()
    tensor<float32> gpu_sigmoid = dnn.sigmoid(neural.track(gpu_activation)).untrack().cpu()
    print(near(cpu_sigmoid[0, 0].item(), gpu_sigmoid[0, 0].item()) and near(cpu_sigmoid[0, 1].item(), gpu_sigmoid[0, 1].item()))

    tensor<float32> cpu_softmax = dnn.softmax(neural.track(cpu_activation)).untrack()
    tensor<float32> gpu_softmax = dnn.softmax(neural.track(gpu_activation)).untrack().cpu()
    print(near(cpu_softmax[0, 0].item(), gpu_softmax[0, 0].item()) and near(cpu_softmax[0, 1].item(), gpu_softmax[0, 1].item()))

    tensor<float32> cpu_gelu = dnn.gelu(neural.track(cpu_activation)).untrack()
    tensor<float32> gpu_gelu = dnn.gelu(neural.track(gpu_activation)).untrack().cpu()
    print(near(cpu_gelu[0, 0].item(), gpu_gelu[0, 0].item()) and near(cpu_gelu[0, 1].item(), gpu_gelu[0, 1].item()))

    tensor<float32> cpu_probability = tensor.zeros<float32>([1, 2])
    cpu_probability[0, 0] = float32(0.25)
    cpu_probability[0, 1] = float32(0.75)
    tensor<float32> gpu_probability = cpu_probability.gpu($GPU_INDEX)
    tensor<float32> cpu_binary_target = tensor.zeros<float32>([1, 2])
    cpu_binary_target[0, 1] = float32(1)
    tensor<float32> gpu_binary_target = cpu_binary_target.gpu($GPU_INDEX)

    float32 cpu_mse_loss = dnn.mse(neural.track(cpu_probability), cpu_binary_target).untrack().item()
    float32 gpu_mse_loss = dnn.mse(neural.track(gpu_probability), gpu_binary_target).untrack().item()
    print(near(cpu_mse_loss, gpu_mse_loss))

    float32 cpu_bce = dnn.binary_cross_entropy(neural.track(cpu_probability), cpu_binary_target).untrack().item()
    float32 gpu_bce = dnn.binary_cross_entropy(neural.track(gpu_probability), gpu_binary_target).untrack().item()
    print(near(cpu_bce, gpu_bce))

    float32 cpu_bce_logits = dnn.binary_cross_entropy_with_logits(neural.track(cpu_activation), cpu_binary_target).untrack().item()
    float32 gpu_bce_logits = dnn.binary_cross_entropy_with_logits(neural.track(gpu_activation), gpu_binary_target).untrack().item()
    print(near(cpu_bce_logits, gpu_bce_logits))

    float32 cpu_cross_entropy = dnn.cross_entropy(neural.track(cpu_activation), cpu_binary_target).untrack().item()
    float32 gpu_cross_entropy = dnn.cross_entropy(neural.track(gpu_activation), gpu_binary_target).untrack().item()
    print(near(cpu_cross_entropy, gpu_cross_entropy))

    LinearModel cpu_model = LinearModel(dense = cpu_linear)
    LinearModel gpu_model = LinearModel(dense = gpu_linear)
    tensor<float32> cpu_target = tensor.zeros<float32>([1, 1])
    tensor<float32> gpu_target = cpu_target.gpu($GPU_INDEX)
    neural.Gradients cpu_grad = neural.grad(
        dnn.mse(cpu_model.dense.forward(neural.track(cpu_samples)), cpu_target)
    )
    neural.Gradients gpu_grad = neural.grad(
        dnn.mse(gpu_model.dense.forward(neural.track(gpu_samples)), gpu_target)
    )
    dnn.SGDOptimizer cpu_sgd = try dnn.SGD(rate = 0.1)
    dnn.SGDOptimizer gpu_sgd = try dnn.SGD(rate = 0.1)
    cpu_sgd.step(&cpu_model, cpu_grad)
    gpu_sgd.step(&gpu_model, gpu_grad)
    print(near(
        cpu_model.dense.weight.raw()[0, 0].item(),
        gpu_model.dense.weight.raw()[0, 0].item()
    ))

    ConvModel cpu_conv_model = ConvModel(convolution = cpu_conv)
    ConvModel gpu_conv_model = ConvModel(convolution = gpu_conv)
    tensor<float32> cpu_zero_image = tensor.zeros<float32>([1, 1, 2, 2])
    tensor<float32> gpu_zero_image = cpu_zero_image.gpu($GPU_INDEX)
    neural.Gradients cpu_conv_grad = neural.grad(
        dnn.mse(cpu_conv_model.convolution.forward(neural.track(cpu_pixels)), cpu_zero_image)
    )
    neural.Gradients gpu_conv_grad = neural.grad(
        dnn.mse(gpu_conv_model.convolution.forward(neural.track(gpu_pixels)), gpu_zero_image)
    )
    dnn.SGDOptimizer cpu_conv_sgd = try dnn.SGD(rate = 0.1)
    dnn.SGDOptimizer gpu_conv_sgd = try dnn.SGD(rate = 0.1)
    cpu_conv_sgd.step(&cpu_conv_model, cpu_conv_grad)
    gpu_conv_sgd.step(&gpu_conv_model, gpu_conv_grad)
    print(near(
        cpu_conv_model.convolution.weight.raw()[0, 0, 0, 0].item(),
        gpu_conv_model.convolution.weight.raw()[0, 0, 0, 0].item()
    ))

    ConvModel cpu_conv3_model = ConvModel(convolution = cpu_conv3)
    ConvModel gpu_conv3_model = ConvModel(convolution = gpu_conv3)
    tensor<float32> cpu_conv3_target = tensor.zeros<float32>([2, 2, 3, 4])
    tensor<float32> gpu_conv3_target = cpu_conv3_target.gpu($GPU_INDEX)
    neural.Gradients cpu_conv3_grad = neural.grad(
        dnn.mse(
            cpu_conv3_model.convolution.forward(neural.track(cpu_conv3_input)),
            cpu_conv3_target
        )
    )
    neural.Gradients gpu_conv3_grad = neural.grad(
        dnn.mse(
            gpu_conv3_model.convolution.forward(neural.track(gpu_conv3_input)),
            gpu_conv3_target
        )
    )
    dnn.SGDOptimizer cpu_conv3_sgd = try dnn.SGD(rate = 0.01)
    dnn.SGDOptimizer gpu_conv3_sgd = try dnn.SGD(rate = 0.01)
    cpu_conv3_sgd.step(&cpu_conv3_model, cpu_conv3_grad)
    gpu_conv3_sgd.step(&gpu_conv3_model, gpu_conv3_grad)
    print(near(
        cpu_conv3_model.convolution.weight.raw()[0, 0, 1, 1].item(),
        gpu_conv3_model.convolution.weight.raw()[0, 0, 1, 1].item()
    ))
    print(near(
        cpu_conv3_model.convolution.bias.raw()[1].item(),
        gpu_conv3_model.convolution.bias.raw()[1].item()
    ))

    NormModel cpu_norm_model = NormModel(normalization = cpu_norm)
    NormModel gpu_norm_model = NormModel(normalization = gpu_norm)
    neural<float32> cpu_norm_train = cpu_norm_model.normalization.forward(neural.track(cpu_norm_input))
    neural<float32> gpu_norm_train = gpu_norm_model.normalization.forward(neural.track(gpu_norm_input))
    neural.Gradients cpu_norm_grad = neural.grad(neural.mean(cpu_norm_train * cpu_norm_train))
    neural.Gradients gpu_norm_grad = neural.grad(neural.mean(gpu_norm_train * gpu_norm_train))
    dnn.SGDOptimizer cpu_norm_sgd = try dnn.SGD(rate = 0.1)
    dnn.SGDOptimizer gpu_norm_sgd = try dnn.SGD(rate = 0.1)
    cpu_norm_sgd.step(&cpu_norm_model, cpu_norm_grad)
    gpu_norm_sgd.step(&gpu_norm_model, gpu_norm_grad)
    print(near(
        cpu_norm_model.normalization.scale.raw()[0].item(),
        gpu_norm_model.normalization.scale.raw()[0].item()
    ))
    tensor<float32> cpu_norm_infer = cpu_norm_model.normalization.infer(cpu_norm_input)
    tensor<float32> gpu_norm_infer = gpu_norm_model.normalization.infer(gpu_norm_input).cpu()
    print(near(cpu_norm_infer[0, 0].item(), gpu_norm_infer[0, 0].item()))

    dnn.LinearLayer cpu_adam_layer = dnn.LinearLayer(
        weight = neural.Parameter<float32>(value = tensor.ones<float32>([1, 1])),
        bias = neural.Parameter<float32>(value = tensor.zeros<float32>([1]))
    )
    dnn.LinearLayer gpu_adam_layer = dnn.LinearLayer(
        weight = neural.Parameter<float32>(value = tensor.ones<float32>([1, 1], gpu = $GPU_INDEX)),
        bias = neural.Parameter<float32>(value = tensor.zeros<float32>([1], gpu = $GPU_INDEX))
    )
    LinearModel cpu_adam_model = LinearModel(dense = cpu_adam_layer)
    LinearModel gpu_adam_model = LinearModel(dense = gpu_adam_layer)
    tensor<float32> cpu_one = tensor.ones<float32>([1, 1])
    tensor<float32> gpu_one = cpu_one.gpu($GPU_INDEX)
    tensor<float32> cpu_zero = tensor.zeros<float32>([1, 1])
    tensor<float32> gpu_zero = cpu_zero.gpu($GPU_INDEX)
    neural.Gradients cpu_adam_grad = neural.grad(
        dnn.mse(cpu_adam_model.dense.forward(neural.track(cpu_one)), cpu_zero)
    )
    neural.Gradients gpu_adam_grad = neural.grad(
        dnn.mse(gpu_adam_model.dense.forward(neural.track(gpu_one)), gpu_zero)
    )
    dnn.AdamOptimizer cpu_adam = try dnn.Adam(rate = 0.1)
    dnn.AdamOptimizer gpu_adam = try dnn.Adam(rate = 0.1)
    cpu_adam.step(&cpu_adam_model, cpu_adam_grad)
    gpu_adam.step(&gpu_adam_model, gpu_adam_grad)
    print(near(
        cpu_adam_model.dense.weight.raw()[0, 0].item(),
        gpu_adam_model.dense.weight.raw()[0, 0].item()
    ))
    print(near(
        cpu_adam_model.dense.bias.raw()[0].item(),
        gpu_adam_model.dense.bias.raw()[0].item()
    ))

    neural.Gradients cpu_adam_grad2 = neural.grad(
        dnn.mse(cpu_adam_model.dense.forward(neural.track(cpu_one)), cpu_zero)
    )
    neural.Gradients gpu_adam_grad2 = neural.grad(
        dnn.mse(gpu_adam_model.dense.forward(neural.track(gpu_one)), gpu_zero)
    )
    cpu_adam.step(&cpu_adam_model, cpu_adam_grad2)
    gpu_adam.step(&gpu_adam_model, gpu_adam_grad2)
    print(near(
        cpu_adam_model.dense.weight.raw()[0, 0].item(),
        gpu_adam_model.dense.weight.raw()[0, 0].item()
    ))
    print(near(
        cpu_adam_model.dense.bias.raw()[0].item(),
        gpu_adam_model.dense.bias.raw()[0].item()
    ))
    return 0

auto result = run()
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

if grep -Fq "GPU 1" <<<"$gpu_info"; then
    gpu1_block="$(awk '
        $0 == "GPU 1" { found = 1; print; next }
        found && /^GPU [0-9]+$/ { exit }
        found { print }
    ' <<<"$gpu_info")"
    if grep -Fq "backend: NVIDIA" <<<"$gpu_block" &&
       grep -Fq "backend: NVIDIA" <<<"$gpu1_block"; then
        cat > "$TMP/dnn-real-nccl.qui" <<'QUI'
import dnn

tensor<float32> first = tensor.ones<float32>([2], gpu = 0)
tensor<float32> second = tensor.ones<float32>([2], gpu = 1) * float32(2)
tensor<float32>[] values = [first, second]
dnn.all_reduce_sum(&values)
tensor<float32> first_cpu = values[0].cpu()
tensor<float32> second_cpu = values[1].cpu()
print(first_cpu[0].item())
print(second_cpu[0].item())
QUI
        nccl_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/dnn-real-nccl.qui")"
        nccl_expected="$(printf '3.0\n3.0')"
        if [[ "$nccl_output" != "$nccl_expected" ]]; then
            echo "DNN NCCL real multi-GPU reduction failed" >&2
            printf '%s\n' "$nccl_output" >&2
            exit 1
        fi
    elif [[ "$REQUIRE_MULTI" == "1" ]]; then
        echo "real DNN multi-GPU integration requires two NVIDIA GPUs" >&2
        exit 1
    fi
elif [[ "$REQUIRE_MULTI" == "1" ]]; then
    echo "real DNN multi-GPU integration requires gpu(1)" >&2
    printf '%s\n' "$gpu_info" >&2
    exit 1
fi

echo "dnn real GPU integration: ok on gpu($GPU_INDEX)"
