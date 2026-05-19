import torch
import torch.nn as nn
from .resnet import BasicBlock, ResNet
from .masking import masking

class ResMasking(ResNet):
    """Residual Masking network for emotion recognition.
    
    This network extends ResNet with attention masking modules applied at each layer.
    The masking modules help focus on important features for emotion classification.

    Args:
        in_channels (int): Number of input channels, default is 3
        num_classes (int): Number of emotion classes, default is 8
        weight_path (str): Path to pre-trained weights, optional
    """
    def __init__(self, in_channels = 3, num_classes=8, weight_path=None):
        super(ResMasking, self).__init__(block=BasicBlock, layers=[3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)
        if weight_path:
            print('weight_path: ', weight_path)
            state_dict = torch.load(weight_path)["net"]
            self.load_state_dict(state_dict)

        self.fc = nn.Linear(512, num_classes)

        self.mask1 = masking(64, 64, depth=4)
        self.mask2 = masking(128, 128, depth=3)
        self.mask3 = masking(256, 256, depth=2)
        self.mask4 = masking(512, 512, depth=1)

    def forward(self, x):
        """Forward pass of the ResMasking network.
        
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
        m = self.mask1(x)
        x = x * (1 + m)

        x = self.layer2(x)
        m = self.mask2(x)
        x = x * (1 + m)

        x = self.layer3(x)
        m = self.mask3(x)
        x = x * (1 + m)

        x = self.layer4(x)
        m = self.mask4(x)
        x = x * (1 + m)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x


def resmasking_dropout(in_channels = 3, num_classes=8, weight_path=None):
    """Create a ResMasking model with dropout layer.

    Args:
        in_channels (int): Number of input channels, default is 3
        num_classes (int): Number of emotion classes, default is 8
        weight_path (str): Path to pre-trained weights, optional

    Returns:
        ResMasking: ResMasking model with dropout applied to the final layer
    """
    model = ResMasking(in_channels = in_channels, num_classes=num_classes, weight_path=weight_path)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model


