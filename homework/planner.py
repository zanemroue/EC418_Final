# File: homework/planner.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: Tensor of shape (batch_size, H, W)
    :return: Tensor of shape (batch_size, 2) with coordinates in (-1, 1)
    """
    batch_size, H, W = logit.shape
    weights = F.softmax(logit.view(batch_size, -1), dim=1).view_as(logit)
    
    pos_x = torch.linspace(-1, 1, W).to(logit.device)
    pos_y = torch.linspace(-1, 1, H).to(logit.device)
    
    expected_x = (weights.sum(1) * pos_x).sum(1)
    expected_y = (weights.sum(2) * pos_y).sum(1)
    
    return torch.stack([expected_x, expected_y], dim=1)

class Planner(nn.Module):
    def __init__(self):
        super(Planner, self).__init__()
        # Load ResNet18 model without pretrained weights
        resnet = models.resnet18(weights=None)
        # Modify the first convolutional layer
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the maxpool layer
        resnet.maxpool = nn.Identity()
        # Remove the avgpool layer
        resnet.avgpool = nn.Identity()
        # Remove the fully connected layer
        resnet.fc = nn.Identity()
        
        # Assign the modified ResNet to self.resnet
        self.resnet = resnet  # Retains original layer names like resnet.conv1, resnet.layer1, etc.
        
        # Add custom convolutional layers for spatial regression
        self.conv_regression = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 2, kernel_size=1)
        )
    
    def forward(self, img):
        # Manually define the forward pass to prevent flattening
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Do not apply avgpool or flatten
        x = self.conv_regression(x)  # x now has shape [batch_size, 2, H, W]
        x = spatial_argmax(x[:, 0])  # x is now [batch_size, 2]
        return x

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError(f"Model type '{type(model)}' not supported!")

def load_model():
    from torch import load
    from os import path
    model = Planner()
    model_path = path.join(path.dirname(path.abspath(__file__)), 'planner.th')
    if path.exists(model_path):
        state_dict = load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"No saved model found at {model_path}")
    return model

if __name__ == '__main__':
    from controller import control
    from utils import PyTux
    from argparse import ArgumentParser

    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)