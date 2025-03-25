import torch
import torch.nn as nn
import torch.nn.functional as F

class _InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(_InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.layers(x)
        else:
            return self.layers(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Modified pooling branch: Use GroupNorm instead of BatchNorm
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),  # Replace BatchNorm with GroupNorm
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        _, _, h, w = x.size()
        
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        
        feat5 = self.pool(x)
        feat5 = F.interpolate(feat5, size=(h, w), mode='bilinear', align_corners=True)
        
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.bottleneck(out)
        
        return out

class MobileNetEncoder(nn.Module):
    def __init__(self, alpha=1.0):
        super(MobileNetEncoder, self).__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, int(32*alpha), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32*alpha)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Inverted residual blocks
        self.block1 = _InvertedResidualBlock(int(32*alpha), int(16*alpha), stride=1, expand_ratio=1)
        
        self.block2 = nn.Sequential(
            _InvertedResidualBlock(int(16*alpha), int(24*alpha), stride=2, expand_ratio=6),
            _InvertedResidualBlock(int(24*alpha), int(24*alpha), stride=1, expand_ratio=6)
        )
        
        self.block3 = nn.Sequential(
            _InvertedResidualBlock(int(24*alpha), int(32*alpha), stride=2, expand_ratio=6),
            _InvertedResidualBlock(int(32*alpha), int(32*alpha), stride=1, expand_ratio=6),
            _InvertedResidualBlock(int(32*alpha), int(32*alpha), stride=1, expand_ratio=6)
        )
        
        self.block4 = nn.Sequential(
            _InvertedResidualBlock(int(32*alpha), int(64*alpha), stride=2, expand_ratio=6),
            _InvertedResidualBlock(int(64*alpha), int(64*alpha), stride=1, expand_ratio=6),
            _InvertedResidualBlock(int(64*alpha), int(64*alpha), stride=1, expand_ratio=6),
            _InvertedResidualBlock(int(64*alpha), int(64*alpha), stride=1, expand_ratio=6)
        )
        
        self.block5 = nn.Sequential(
            _InvertedResidualBlock(int(64*alpha), int(96*alpha), stride=1, expand_ratio=6),
            _InvertedResidualBlock(int(96*alpha), int(96*alpha), stride=1, expand_ratio=6),
            _InvertedResidualBlock(int(96*alpha), int(96*alpha), stride=1, expand_ratio=6)
        )
        
        self.block6 = nn.Sequential(
            _InvertedResidualBlock(int(96*alpha), int(160*alpha), stride=2, expand_ratio=6),
            _InvertedResidualBlock(int(160*alpha), int(160*alpha), stride=1, expand_ratio=6),
            _InvertedResidualBlock(int(160*alpha), int(160*alpha), stride=1, expand_ratio=6)
        )
        
        self.block7 = _InvertedResidualBlock(int(160*alpha), int(320*alpha), stride=1, expand_ratio=6)
        
    def forward(self, x):
        features = {}
        
        x = self.conv1(x)
        x = self.block1(x)
        
        x = self.block2(x)
        features["low_level"] = x  # 1/4 size feature map for skip connection
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        
        features["high_level"] = x  # 1/16 size feature map for ASPP
        
        return features

class POSALGenerator(nn.Module):
    def __init__(self, num_classes=2):
        super(POSALGenerator, self).__init__()
        
        # Encoder (MobileNetV2)
        self.encoder = MobileNetEncoder(alpha=1.0)
        
        # ASPP module
        self.aspp = ASPP(320, 256)
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256+48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        # Get original dimensions for upsampling
        input_shape = x.shape[-2:]
        
        # Encoder features
        features = self.encoder(x)
        
        # ASPP
        x = self.aspp(features["high_level"])
        
        # Decoder with skip connection
        low_level_feat = self.low_level_conv(features["low_level"])
        
        # Upsample ASPP features to match low-level feature size
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        
        # Concatenate features
        x = torch.cat((x, low_level_feat), dim=1)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to original image size
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        
        return x

class POSALDiscriminator(nn.Module):
    def __init__(self, num_classes=2):
        super(POSALDiscriminator, self).__init__()
        
        # Input is segmentation masks with num_classes channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_classes, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x

class POSAL(nn.Module):
    """
    Patch-based Output Space Adversarial Learning for Joint Optic Disc and Cup Segmentation.
    """
    def __init__(self, num_classes=2):
        super(POSAL, self).__init__()
        self.num_classes = num_classes
        self.generator = POSALGenerator(num_classes)
        self.discriminator = POSALDiscriminator(num_classes)
        
    def forward(self, x):
        # During inference, we only use the generator
        return self.generator(x)

class POSALModel:
    """
    POSALModel class following the pattern of other model classes in the project.
    """
    def __init__(self, in_channels=3, out_channels=3, pretrained=False, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pretrained = pretrained
        self.kwargs = kwargs
        
    def __call__(self, **kwargs):
        """
        Make this class callable to support the model initialization pattern.
        This allows it to be used like a constructor when called from initialize_model.
        """
        # Create a new instance with proper parameter handling
        return POSALModel(
            in_channels=kwargs.get('in_channels', self.in_channels),
            out_channels=kwargs.get('out_channels', self.out_channels),
            pretrained=kwargs.get('pretrained', self.pretrained),
            **{k: v for k, v in kwargs.items() if k not in ['in_channels', 'out_channels', 'pretrained']}
        )
        
    @staticmethod
    def get_default_config(image_size):
        """
        Get default configuration for POSAL model.
        
        Args:
            image_size: Size of input image (for consistent interface)
            
        Returns:
            Dictionary with default configuration
        """
        return {
            "in_channels": 3,
            "out_channels": 3,
            "feature_size": 48,
            "dropout_rate": 0.1,
            "norm_name": "instance",
            "use_adversarial": True,
            "adversarial_weight": 0.01,
            "alpha": 1.0  # MobileNet width multiplier
        }
    
    def build(self):
        """
        Build the POSAL model with the specified configuration.
        
        Returns:
            POSAL model instance
        """
        model = POSAL(num_classes=self.out_channels)
        # Make sure model is in proper state for both training and inference
        model.train()
        return model