import torch
import torch.nn as nn
import timm

class PlantClassifier(nn.Module):
    def __init__(self, num_classes=38, backbone="efficientnetv2_rw_s", dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        if hasattr(self.backbone, "classifier"):
            n = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(n, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )
        elif hasattr(self.backbone, "fc"):
            n = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(n, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes),
            )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for m in [self.backbone.classifier] if hasattr(self.backbone, "classifier") else [self.backbone.fc]:
            for p in m.parameters():
                p.requires_grad = True

    def unfreeze_last_fraction(self, fraction=0.3):
        mods = list(self.backbone.children())
        k = int(len(mods) * (1 - fraction))
        for i, m in enumerate(mods):
            req = i >= k
            for p in m.parameters():
                p.requires_grad = req
