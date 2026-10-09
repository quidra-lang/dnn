# Quidra DNN

Quidra DNN is the first-party deep-model architecture package, imported as `dnn`.

Dependency layers:

```text
Layer 1: Core
Layer 2: Math
Layer 3: NN / Vision / Video
Layer 4: DNN
```

Layer numbers constrain dependency direction; they do not imply dependencies
on every lower-layer sibling. DNN currently depends on Core, Math, and NN; it
does not require Vision or Video.

DNN owns deep-network/model architecture semantics. Reusable neural-network building blocks are not implemented here: `Parameter`, `State`, `FC`, `Conv2D`, `BatchNorm`, `Dropout`, activations, losses, optimizers, persistence, collectives, cuDNN/NCCL integration, and NN compiler fusion live in the first-party `nn` package.

DNN errors carry a stable code while preserving their operation-specific
messages: `DNN_ARGUMENT` covers invalid architecture options (such as a
nonpositive block count or unsupported shortcut stride), and `DNN_SHAPE`
covers incompatible input ranks, channels and spatial dimensions.

DNN currently provides paper-oriented ImageNet architecture families under `dnn.model`:

- `dnn.model.AlexNet`
- `dnn.model.VGGA`, `VGGALRN`, `VGGB`, `VGGC`, `VGGD`, `VGGE`
- `dnn.model.VGG11`, `VGG13`, `VGG16`, `VGG19`
- `dnn.model.ResNet18`, `ResNet34`, `ResNet50`, `ResNet101`, `ResNet152`

The implementations compose `nn.*` layers/operations and `math.*` numerical semantics. DNN has no package-native kernel, accelerator-library loader, optimizer semantics, or compiler-extension descriptor of its own.

```quidra
import dnn
import nn

dnn.model.ResNet18 model = dnn.model.ResNet18()
nn.Adam optimizer = nn.Adam()
```

Core remains unaware of both NN and DNN operation names. Math owns generic numerical semantics; NN owns generic neural-network semantics and acceleration policy; DNN owns architecture composition.
