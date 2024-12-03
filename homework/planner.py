# File: homework/planner.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack((
        (weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)
    ), 1)

class Planner(nn.Module):
    def __init__(self):
        super(Planner, self).__init__()
        # Load ResNet18 model
        self.resnet = models.resnet18(pretrained=False)
        # Modify the first convolutional layer
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        # Replace the fully connected layer to output feature maps for spatial_argmax
        self.resnet.fc = nn.Sequential(
            nn.Conv2d(self.resnet.fc.in_features, 2, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, img):
        """
        Predict the aim point in image coordinate, given the SuperTuxKart image
        @img: (B,3,96,128)
        return: (B,2)
        """
        x = self.resnet(img)  # Output shape: (B, 2, H, W)
        x = spatial_argmax(x[:, 0])
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
        model.load_state_dict(load(model_path, map_location='cpu'))
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