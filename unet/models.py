import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from icecream import ic


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.skip_connection = nn.Identity()
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip_connection(x)
        out = self.conv_path(x)
        out += residual
        out = self.relu(out)
        return out
    
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, alpha=1):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid() # Sigmoid activation for attention map
        )
        self.relu = nn.ReLU(inplace=True)
        self.alpha = alpha

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_in = self.relu(g1 + x1)
        psi_out = self.psi(psi_in)

        if x.size()[2:] != psi_out.size()[2:]:
            psi_out = F.interpolate(psi_out, size=x.size()[2:], mode='bilinear', align_corners=True)

        return self.alpha * x * psi_out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, use_skip_connections=True, use_residual_block=False, use_gated_attention=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_skip_connections = use_skip_connections
        if use_residual_block:
            self.conv_block = ResidualBlock
        else:
            self.conv_block = DoubleConv
        self.use_gated_attention = use_gated_attention
        if self.use_gated_attention and not self.use_skip_connections:
            raise ValueError("Attention Gates require Skip Connections to be enabled.")
        self.inc = self.conv_block(n_channels, 64)  # 256x256x64

        self.pool1 = nn.MaxPool2d(2)  # 256 -> 128
        self.pool2 = nn.MaxPool2d(2)  # 128 -> 64
        self.pool3 = nn.MaxPool2d(2)  # 64 -> 32
        self.pool4 = nn.MaxPool2d(2)  # 32 -> 16

        self.conv_down1 = self.conv_block(64, 128)   # 128x128x128
        self.conv_down2 = self.conv_block(128, 256)  # 64x64x256
        self.conv_down3 = self.conv_block(256, 512)  # 32x32x512
        # 16x16x1024 (after pooling)
        self.conv_bottleneck = self.conv_block(512, 1024)

        self.up1 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2)  # 16 -> 32
        self.up2 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2)  # 32 -> 64
        self.up3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2)  # 64 -> 128
        self.up4 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2)   # 128 -> 256
        
        if self.use_gated_attention:
            self.attn1 = AttentionGate(F_g=64,  F_l=64,  F_int=32)
            self.attn2 = AttentionGate(F_g=128, F_l=128, F_int=64)
            self.attn3 = AttentionGate(F_g=256, F_l=256, F_int=128)
            self.attn4 = AttentionGate(F_g=512, F_l=512, F_int=256)

        up1_in_channels = 512 + 512 if self.use_skip_connections else 512
        up2_in_channels = 256 + 256 if self.use_skip_connections else 256
        up3_in_channels = 128 + 128 if self.use_skip_connections else 128
        up4_in_channels = 64 + 64 if self.use_skip_connections else 64
        
        self.conv_up1 = self.conv_block(up1_in_channels, 512)
        self.conv_up2 = self.conv_block(up2_in_channels, 256)
        self.conv_up3 = self.conv_block(up3_in_channels, 128)
        self.conv_up4 = self.conv_block(up4_in_channels, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)  # 1x1 convolution

    def forward(self, x):
        x1 = self.inc(x)          # 256x256x64 (skip connection 1)
        p1 = self.pool1(x1)       # 128x128x64
        x2 = self.conv_down1(p1)  # 128x128x128 (skip connection 2)
        p2 = self.pool2(x2)       # 64x64x128
        x3 = self.conv_down2(p2)  # 64x64x256 (skip connection 3)
        p3 = self.pool3(x3)       # 32x32x256
        x4 = self.conv_down3(p3)  # 32x32x512 (skip connection 4)
        p4 = self.pool4(x4)       # 16x16x512

        bottleneck = self.conv_bottleneck(p4)  # 16x16x1024

        up1_out = self.up1(bottleneck)        # 32x32x512
        # Concat skip x4 (32x32x512) -> 32x32x1024
        # cat1 = torch.cat([up1_out, x4], dim=1) if self.use_skip_connections else up1_out
        if self.use_skip_connections:
            skip4 = x4
            if self.use_gated_attention:
                skip4 = self.attn4(g=up1_out, x=skip4)
            cat1 = torch.cat([skip4, up1_out], dim=1)
        else:
            cat1 = up1_out
        dec1 = self.conv_up1(cat1)            # 32x32x512

        up2_out = self.up2(dec1)              # 64x64x256
        # Concat skip x3 (64x64x256) -> 64x64x512
        # cat2 = torch.cat([up2_out, x3], dim=1) if self.use_skip_connections else up2_out
        if self.use_skip_connections:
            skip3 = x3
            if self.use_gated_attention:
                skip3 = self.attn3(g=up2_out, x=skip3)
            cat2 = torch.cat([skip3, up2_out], dim=1)
        else:
            cat2 = up2_out
        dec2 = self.conv_up2(cat2)            # 64x64x256

        up3_out = self.up3(dec2)              # 128x128x128
        # Concat skip x2 (128x128x128) -> 128x128x256
        # cat3 = torch.cat([up3_out, x2], dim=1) if self.use_skip_connections else up3_out
        if self.use_skip_connections:
            skip2 = x2
            if self.use_gated_attention:
                skip2 = self.attn2(g=up3_out, x=skip2)
            cat3 = torch.cat([skip2, up3_out], dim=1)
        else:
            cat3 = up3_out
        dec3 = self.conv_up3(cat3)            # 128x128x128

        up4_out = self.up4(dec3)              # 256x256x64
        # Concat skip x1 (256x256x64) -> 256x256x128
        # cat4 = torch.cat([up4_out, x1], dim=1) if self.use_skip_connections else up4_out
        if self.use_skip_connections:
            skip1 = x1
            if self.use_gated_attention:
                skip1 = self.attn1(g=up4_out, x=skip1)
            cat4 = torch.cat([skip1, up4_out], dim=1)
        else:
            cat4 = up4_out
        dec4 = self.conv_up4(cat4)            # 256x256x64

        logits = self.outc(dec4)              # 256x256xN_CLASSES
        return logits
