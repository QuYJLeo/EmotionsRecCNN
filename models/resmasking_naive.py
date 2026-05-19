import copy
import torch
import torch.nn as nn

from .utils import load_state_dict_from_url
from .resnet import BasicBlock, Bottleneck, ResNet, resnet18


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


from .masking import masking


class ResMaskingNaive(ResNet):
    """Naive Residual Masking network with pre-trained ResNet34 weights.
    
    This network extends ResNet34 with masking modules, but the masks are not applied
    in the forward pass (naive version). It's used for comparison or baseline purposes.

    Args:
        weight_path (str): Path placeholder (not used, included for interface consistency)
    """
    def __init__(self, weight_path):
        super(ResMaskingNaive, self).__init__(
            block=BasicBlock, layers=[3, 4, 6, 3], in_channels=3, num_classes=1000
        )
        state_dict = load_state_dict_from_url(model_urls["resnet34"], progress=True)
        self.load_state_dict(state_dict)

        self.fc = nn.Linear(512, 7)

        self.mask1 = masking(64, 64, depth=4)
        self.mask2 = masking(128, 128, depth=3)
        self.mask3 = masking(256, 256, depth=2)
        self.mask4 = masking(512, 512, depth=1)

    def forward(self, x):
        """Forward pass of the ResMaskingNaive network.
        
        Note: Masks are defined but not applied in this naive version.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output logits for emotion classification
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def resmasking_naive_dropout1(in_channels=3, num_classes=7, weight_path=""):
    """Create a ResMaskingNaive model with dropout layer.

    Args:
        in_channels (int): Number of input channels, default is 3
        num_classes (int): Number of emotion classes, default is 7
        weight_path (str): Path placeholder for pre-trained weights

    Returns:
        ResMaskingNaive: ResMaskingNaive model with dropout applied
    """
    model = ResMaskingNaive(weight_path)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, 7)
    )
    return model
