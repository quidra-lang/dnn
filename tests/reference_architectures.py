#!/usr/bin/env python3
from pathlib import Path
import re
import sys

source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1] / "main.qui"
source = source_path.read_text(encoding="utf-8")


def fail(message: str) -> None:
    raise SystemExit(f"reference architecture contract failed: {message}")


def class_body(name: str) -> str:
    match = re.search(
        rf"(?ms)^class {re.escape(name)}\n(.*?)(?=^class |\Z)",
        source,
    )
    if not match:
        fail(f"missing class {name}")
    return match.group(1)


def require(body: str, needle: str, context: str) -> None:
    if needle not in body:
        fail(f"{context}: missing {needle!r}")



# Complete reference architectures live under the exported model namespace.
for name in (
    "AlexNet", "VGGA", "VGGALRN", "VGGB", "VGGC", "VGGD", "VGGE",
    "VGG11", "VGG13", "VGG16", "VGG19",
    "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152",
):
    if re.search(rf"(?m)^class {re.escape(name)}$", source):
        fail(f"reference model leaked into dnn root namespace: {name}")
    if not re.search(rf"(?m)^class model\.{re.escape(name)}$", source):
        fail(f"missing model namespace declaration for {name}")

for name in (
    "ImageNetFC4096",
    "VGGBlock1", "VGGBlock2", "VGGBlock3", "VGGBlockC", "VGGBlock4",
    "ResNetStem", "ResNetBasicBlock", "ResNetBottleneckBlock",
    "ResNetBottleneckProjectionBlock", "ResNetBasicIdentityStage",
    "ResNetBasicOptionABlock", "ResNetBasicOptionAStage", "ResNetBottleneckStage",
):
    if re.search(rf"(?m)^class {re.escape(name)}$", source):
        fail(f"reference helper leaked into dnn root namespace: {name}")
    if not re.search(rf"(?m)^class model\.{re.escape(name)}$", source):
        fail(f"missing model namespace helper declaration for {name}")

# Paper initialization paths.
for needle in (
    "tensor<float32> | error normal_weights(",
    "math.sqrt(-2.0 * math.log(radius_uniform))",
    "math.cos(2.0 * math.pi * angle_uniform)",
    "FC | error normal_fc(",
    "Conv2D | error normal_conv2d(",
):
    require(source, needle, "paper initializer")


# model.AlexNet: five learned convolution layers, original two-GPU connectivity,
# 4096/4096/1000 fully-connected widths, LRN, and overlapping 3x3/2 pooling.
alex = class_body("model.AlexNet")
if len(re.findall(r"(?m)^    Conv2D conv[1-5]$", alex)) != 5:
    fail("model.AlexNet must expose exactly five convolution fields")
for needle in (
    "normal_conv2d(",
    "3, 96, 11, 0.01",
    "96, 256, 5, 0.01",
    "bias_value = 1.0, padding = 2",
    "seed = seed + 1, groups = 2",
    "256, 384, 3, 0.01",
    "bias_value = 0.0, padding = 1",
    "384, 384, 3, 0.01",
    "seed = seed + 3, groups = 2",
    "384, 256, 3, 0.01",
    "seed = seed + 4, groups = 2",
    "model.ImageNetFC4096(",
    "9216, seed + 5",
    "weight_standard_deviation = 0.01",
    "hidden_bias = 1.0",
    "internal.require_imagenet_input(value, 224)",
):
    require(alex, needle, "model.AlexNet")
if alex.count("local_response_normalization(") != 2:
    fail("model.AlexNet must contain exactly two LRN applications")
if alex.count("max_pool2d(output, 3, 2)") != 3:
    fail("model.AlexNet must contain exactly three overlapping 3x3 stride-2 max pools")

classifier = class_body("model.ImageNetFC4096")
for needle in (
    "normal_fc(",
    "features_in, 4096, weight_standard_deviation",
    "4096, 4096, weight_standard_deviation",
    "4096, 1000, weight_standard_deviation",
    "Dropout(0.5",
    "inverted = false",
):
    require(classifier, needle, "4096/4096/1000 classifier")

# VGG Table 1, including the A-LRN and C configurations rather than only the
# later VGG-16/VGG-19 shorthand.
vgg_fields = {
    "model.VGGA": ["model.VGGBlock1", "model.VGGBlock1", "model.VGGBlock2", "model.VGGBlock2", "model.VGGBlock2"],
    "model.VGGALRN": ["model.VGGBlock1", "model.VGGBlock1", "model.VGGBlock2", "model.VGGBlock2", "model.VGGBlock2"],
    "model.VGGB": ["model.VGGBlock2", "model.VGGBlock2", "model.VGGBlock2", "model.VGGBlock2", "model.VGGBlock2"],
    "model.VGGC": ["model.VGGBlock2", "model.VGGBlock2", "model.VGGBlockC", "model.VGGBlockC", "model.VGGBlockC"],
    "model.VGGD": ["model.VGGBlock2", "model.VGGBlock2", "model.VGGBlock3", "model.VGGBlock3", "model.VGGBlock3"],
    "model.VGGE": ["model.VGGBlock2", "model.VGGBlock2", "model.VGGBlock4", "model.VGGBlock4", "model.VGGBlock4"],
}
for name, blocks in vgg_fields.items():
    body = class_body(name)
    actual = []
    for index in range(1, 6):
        match = re.search(rf"(?m)^    ([A-Za-z_][A-Za-z0-9_.]*) block{index}$", body)
        if not match:
            fail(f"{name}: missing block{index}")
        actual.append(match.group(1))
    if actual != blocks:
        fail(f"{name}: blocks {actual!r} != {blocks!r}")
    require(body, "model.ImageNetFC4096(", name)
    require(body, "25088, seed +", name)
    require(body, "internal.require_imagenet_input(value, 224)", name)

if class_body("model.VGGALRN").count("local_response_normalization(") != 1:
    fail("VGG A-LRN must contain exactly one LRN after the first convolution block")
for name in ("model.VGGA", "model.VGGB", "model.VGGC", "model.VGGD", "model.VGGE"):
    if "local_response_normalization(" in class_body(name):
        fail(f"{name} must not contain LRN")

block_c = class_body("model.VGGBlockC")
for needle in (
    "normal_conv2d(",
    "channels_in, channels_out, 3, math.sqrt(0.01)",
    "channels_out, channels_out, 3, math.sqrt(0.01)",
    "channels_out, channels_out, 1, math.sqrt(0.01)",
):
    require(block_c, needle, "VGG configuration C block")

for name in ("model.VGGBlock1", "model.VGGBlock2", "model.VGGBlock3", "model.VGGBlockC", "model.VGGBlock4"):
    body = class_body(name)
    require(body, "normal_conv2d(", f"{name} paper initialization")
    require(body, "math.sqrt(0.01)", f"{name} paper variance")

for name in ("model.VGGA", "model.VGGALRN", "model.VGGB", "model.VGGC", "model.VGGD", "model.VGGE"):
    require(
        class_body(name),
        "weight_standard_deviation = math.sqrt(0.01)",
        f"{name} classifier paper variance",
    )

for name in ("model.VGGB", "model.VGGC", "model.VGGD", "model.VGGE"):
    body = class_body(name)
    require(body, "void initialize_from(model.VGGA source)", f"{name} paper pre-initialization")
    for needle in (
        "block1.conv1.weight.replace(source.block1.conv1.weight.raw())",
        "block2.conv1.weight.replace(source.block2.conv1.weight.raw())",
        "block3.conv1.weight.replace(source.block3.conv1.weight.raw())",
        "block3.conv2.weight.replace(source.block3.conv2.weight.raw())",
        "classifier.fc1.weight.replace(source.classifier.fc1.weight.raw())",
        "classifier.fc2.weight.replace(source.classifier.fc2.weight.raw())",
        "classifier.fc3.weight.replace(source.classifier.fc3.weight.raw())",
    ):
        require(body, needle, f"{name} net-A transfer")

aliases = {"model.VGG11": "model.VGGA", "model.VGG13": "model.VGGB", "model.VGG16": "model.VGGD", "model.VGG19": "model.VGGE"}
for alias, target in aliases.items():
    require(class_body(alias), f"{target} network", alias)

for alias in ("model.VGG13", "model.VGG16", "model.VGG19"):
    require(
        class_body(alias),
        "void initialize_from(model.VGG11 source)",
        f"{alias} net-A transfer alias",
    )

# ResNet Table 1. 18/34 use the paper's parameter-free option-A shortcuts for
# stage transitions; 50/101/152 use option-B projections and v1 stride
# placement in the first 1x1 bottleneck convolution. Stage repetition is a
# runtime array count so one stage implementation covers every paper depth.
identity_stage = class_body("model.ResNetBasicIdentityStage")
for needle in (
    "model.ResNetBasicBlock[] blocks",
    "for index in range(count)",
    "seed + index * 2",
    "for block in blocks",
):
    require(identity_stage, needle, "ResNet basic identity stage")

option_a_stage = class_body("model.ResNetBasicOptionAStage")
for needle in (
    "model.ResNetBasicOptionABlock first",
    "model.ResNetBasicBlock[] blocks",
    "for index in range(count - 1)",
    "seed + 2 + index * 2",
    "for block in blocks",
):
    require(option_a_stage, needle, "ResNet basic option-A stage")

bottleneck_stage = class_body("model.ResNetBottleneckStage")
for needle in (
    "model.ResNetBottleneckProjectionBlock first",
    "model.ResNetBottleneckBlock[] blocks",
    "for index in range(count - 1)",
    "seed + 4 + index * 3",
    "for block in blocks",
):
    require(bottleneck_stage, needle, "ResNet bottleneck stage")

for old_stage in (
    "ResNetBasicIdentityStage2",
    "ResNetBasicIdentityStage3",
    "ResNetBasicOptionAStage2",
    "ResNetBasicOptionAStage3",
    "ResNetBasicOptionAStage4",
    "ResNetBasicOptionAStage6",
    "ResNetBasicProjectionStage2",
    "ResNetBasicProjectionStage3",
    "ResNetBasicProjectionStage4",
    "ResNetBasicProjectionStage6",
    "ResNetBottleneckStage3",
    "ResNetBottleneckStage4",
    "ResNetBottleneckStage6",
    "ResNetBottleneckStage8",
    "ResNetBottleneckStage23",
    "ResNetBottleneckStage36",
):
    if re.search(rf"(?m)^class (?:model\.)?{re.escape(old_stage)}$", source):
        fail(f"obsolete expanded stage class remains: {old_stage}")

basic_models = {
    "model.ResNet18": (
        ("model.ResNetBasicIdentityStage", "conv2_x", "model.ResNetBasicIdentityStage(64, 2, seed + 10)"),
        ("model.ResNetBasicOptionAStage", "conv3_x", "model.ResNetBasicOptionAStage(64, 128, 2, seed + 30)"),
        ("model.ResNetBasicOptionAStage", "conv4_x", "model.ResNetBasicOptionAStage(128, 256, 2, seed + 60)"),
        ("model.ResNetBasicOptionAStage", "conv5_x", "model.ResNetBasicOptionAStage(256, 512, 2, seed + 100)"),
    ),
    "model.ResNet34": (
        ("model.ResNetBasicIdentityStage", "conv2_x", "model.ResNetBasicIdentityStage(64, 3, seed + 10)"),
        ("model.ResNetBasicOptionAStage", "conv3_x", "model.ResNetBasicOptionAStage(64, 128, 4, seed + 30)"),
        ("model.ResNetBasicOptionAStage", "conv4_x", "model.ResNetBasicOptionAStage(128, 256, 6, seed + 60)"),
        ("model.ResNetBasicOptionAStage", "conv5_x", "model.ResNetBasicOptionAStage(256, 512, 3, seed + 100)"),
    ),
}
for name, stages in basic_models.items():
    body = class_body(name)
    for stage_type, field, construction in stages:
        require(body, f"{stage_type} {field}", name)
        require(body, construction, name)
    require(body, "normal_fc(", name)
    require(body, "512, 1000", name)
    require(body, "internal.require_imagenet_input(value, 224)", name)

bottleneck_models = {
    "model.ResNet50": (
        "model.ResNetBottleneckStage(64, 64, 256, 3, 1, seed + 10)",
        "model.ResNetBottleneckStage(256, 128, 512, 4, 2, seed + 30)",
        "model.ResNetBottleneckStage(512, 256, 1024, 6, 2, seed + 70)",
        "model.ResNetBottleneckStage(1024, 512, 2048, 3, 2, seed + 190)",
    ),
    "model.ResNet101": (
        "model.ResNetBottleneckStage(64, 64, 256, 3, 1, seed + 10)",
        "model.ResNetBottleneckStage(256, 128, 512, 4, 2, seed + 30)",
        "model.ResNetBottleneckStage(512, 256, 1024, 23, 2, seed + 70)",
        "model.ResNetBottleneckStage(1024, 512, 2048, 3, 2, seed + 190)",
    ),
    "model.ResNet152": (
        "model.ResNetBottleneckStage(64, 64, 256, 3, 1, seed + 10)",
        "model.ResNetBottleneckStage(256, 128, 512, 8, 2, seed + 30)",
        "model.ResNetBottleneckStage(512, 256, 1024, 36, 2, seed + 70)",
        "model.ResNetBottleneckStage(1024, 512, 2048, 3, 2, seed + 190)",
    ),
}
for name, constructions in bottleneck_models.items():
    body = class_body(name)
    for field in ("conv2_x", "conv3_x", "conv4_x", "conv5_x"):
        require(body, f"model.ResNetBottleneckStage {field}", name)
    for construction in constructions:
        require(body, construction, name)
    require(body, "normal_fc(", name)
    require(body, "2048, 1000", name)
    require(body, "internal.require_imagenet_input(value, 224)", name)


projection = class_body("model.ResNetBottleneckProjectionBlock")
require(projection, "normal_conv2d(", "ResNet-v1 bottleneck initializer")
require(projection, "channels_in, bottleneck, 1", "ResNet-v1 bottleneck")
require(projection, "stride = stride, seed = seed", "ResNet-v1 bottleneck")
require(projection, "channels_in, channels_out, 1", "ResNet-v1 shortcut")
if "bottleneck, bottleneck, 3,\n            math.sqrt(2.0 / float(bottleneck * 9)),\n            stride = stride" in projection:
    fail("ResNet-v1 must not use the later v1.5 stride placement")

stem = class_body("model.ResNetStem")
for needle in (
    "normal_conv2d(",
    "3, 64, 7, math.sqrt(2.0 / 147.0)",
    "max_pool2d(output, 3, 2, padding = 1)",
):
    require(stem, needle, "ResNet stem")

for name in (
    "model.ResNetStem",
    "model.ResNetBasicBlock",
    "model.ResNetBottleneckBlock",
    "model.ResNetBottleneckProjectionBlock",
    "model.ResNetBasicOptionABlock",
):
    body = class_body(name)
    require(body, "normal_conv2d(", f"{name} He initialization")
    require(body, "math.sqrt(2.0 /", f"{name} He variance")

for name, fan_in in (
    ("model.ResNet18", "512.0"),
    ("model.ResNet34", "512.0"),
    ("model.ResNet50", "2048.0"),
    ("model.ResNet101", "2048.0"),
    ("model.ResNet152", "2048.0"),
):
    body = class_body(name)
    require(body, "normal_fc(", f"{name} final FC He initialization")
    require(body, f"math.sqrt(2.0 / {fan_in})", f"{name} final FC He variance")


print("reference architecture source contracts: ok")
