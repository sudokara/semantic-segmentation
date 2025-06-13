from data import *


def get_vgg_backbone(backbone_name="vgg19", pretrained=True, freeze_backbone=True):
    """Loads a VGG backbone, sets requires_grad, and identifies pool layer indices."""
    if backbone_name == "vgg16":
        vgg = vgg16(pretrained=pretrained)
    elif backbone_name == "vgg19":
        vgg = vgg19(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    features = vgg.features
    pool_indices = [
        i for i, layer in enumerate(features) if isinstance(layer, nn.MaxPool2d)
    ]

    if len(pool_indices) < 5:
        raise ValueError(
            f"{backbone_name} features does not have enough MaxPool2d layers for FCN-8s (needs 5)"
        )

    pool3_idx = pool_indices[2]
    pool4_idx = pool_indices[3]
    pool5_idx = pool_indices[4]

    for param in features.parameters():
        param.requires_grad = not freeze_backbone

    return features, pool3_idx, pool4_idx, pool5_idx


class FCNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.classifier(x)


class FCN32s(nn.Module):
    def __init__(
        self, num_classes, backbone_name="vgg19", pretrained=True, freeze_backbone=True
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone, self.pool3_idx, self.pool4_idx, self.pool5_idx = (
            get_vgg_backbone(backbone_name, pretrained, freeze_backbone)
        )
        self.pool5_channels = 512

        self.head = FCNHead(self.pool5_channels, num_classes)

        self.upsample32 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, stride=32, padding=16
        )

    def _run_backbone(self, x):
        """Runs input through the backbone, capturing features at pool points."""
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == self.pool5_idx:
                return x
        raise RuntimeError("Failed to reach pool5 layer in backbone.")

    def forward(self, x):
        pool5_feat = self._run_backbone(x)
        score = self.head(pool5_feat)
        upscore = self.upsample32(score)
        return upscore


class FCN16s(nn.Module):
    def __init__(
        self, num_classes, backbone_name="vgg19", pretrained=True, freeze_backbone=True
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone, self.pool3_idx, self.pool4_idx, self.pool5_idx = (
            get_vgg_backbone(backbone_name, pretrained, freeze_backbone)
        )
        self.pool4_channels = 512
        self.pool5_channels = 512

        self.score_pool5 = FCNHead(self.pool5_channels, num_classes)
        self.score_pool4 = nn.Conv2d(
            self.pool4_channels, num_classes, kernel_size=1)

        self.upsample2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        )
        self.upsample16 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=32, stride=16, padding=8
        )

    def _run_backbone(self, x):
        """Runs input through the backbone, capturing features at pool points."""
        features = {}
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == self.pool4_idx:
                features["pool4"] = x
            elif i == self.pool5_idx:
                features["pool5"] = x
        if "pool4" not in features or "pool5" not in features:
            raise RuntimeError(
                "Failed to capture required pool4/pool5 features.")
        return features

    def forward(self, x):
        features = self._run_backbone(x)
        pool4_feat = features["pool4"]
        pool5_feat = features["pool5"]

        score_p5 = self.score_pool5(pool5_feat)
        upscore2 = self.upsample2(score_p5)

        score_p4 = self.score_pool4(pool4_feat)

        fused_score = upscore2 + score_p4
        upscore16 = self.upsample16(fused_score)
        return upscore16


class FCN8s(nn.Module):
    def __init__(
        self, num_classes, backbone_name="vgg19", pretrained=True, freeze_backbone=True
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone, self.pool3_idx, self.pool4_idx, self.pool5_idx = (
            get_vgg_backbone(backbone_name, pretrained, freeze_backbone)
        )
        self.pool3_channels = 256
        self.pool4_channels = 512
        self.pool5_channels = 512

        self.score_pool5 = FCNHead(self.pool5_channels, num_classes)
        self.score_pool4 = nn.Conv2d(
            self.pool4_channels, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(
            self.pool3_channels, num_classes, kernel_size=1)

        self.upsample2a = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        )
        self.upsample2b = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1
        )
        self.upsample8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4
        )

    def _run_backbone(self, x):
        """Runs input through the backbone, capturing features at pool points."""
        features = {}
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == self.pool3_idx:
                features["pool3"] = x
            elif i == self.pool4_idx:
                features["pool4"] = x
            elif i == self.pool5_idx:
                features["pool5"] = x
        if (
            "pool3" not in features
            or "pool4" not in features
            or "pool5" not in features
        ):
            raise RuntimeError(
                "Failed to capture required pool3/pool4/pool5 features.")
        return features

    def forward(self, x):
        features = self._run_backbone(x)
        pool3_feat = features["pool3"]
        pool4_feat = features["pool4"]
        pool5_feat = features["pool5"]

        score_p5 = self.score_pool5(pool5_feat)
        upscore_p5_x2 = self.upsample2a(score_p5)

        score_p4 = self.score_pool4(pool4_feat)

        fuse_p4_p5 = score_p4 + upscore_p5_x2
        upscore_p45_x2 = self.upsample2b(fuse_p4_p5)

        score_p3 = self.score_pool3(pool3_feat)

        fuse_p3_p4_p5 = score_p3 + upscore_p45_x2
        upscore8 = self.upsample8(fuse_p3_p4_p5)

        return upscore8
