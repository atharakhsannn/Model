import torch
import torch.nn as nn
from torchvision import models

class DensityMapRegressor(nn.Module):

    def __init__(self, pretrained=True):
        super(DensityMapRegressor, self).__init__()

        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        self.features = mobilenet.features

        self.dilated_convs = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.features(x)

        x = self.dilated_convs(x)

        x = self.upsample(x)

        return x

if __name__ == '__main__':
    model = DensityMapRegressor(pretrained=True)
    print(model)

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output min:   {output.min().item():.6f}")
    print(f"Output max:   {output.max().item():.6f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
