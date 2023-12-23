import torch
import torch.nn as nn
import torch.nn.functional as F
from util import torchutils
from torchvision.models import vgg16, VGG19_Weights

def vgg():
    model = vgg16()
    checkpoint = torch.hub.load_state_dict_from_url(
        url="https://download.pytorch.org/models/vgg16-397923af.pth", # url="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        map_location="cpu", check_hash=True
    )
    checkpoint.pop('classifier.0.weight')
    checkpoint.pop('classifier.0.bias')
    checkpoint.pop('classifier.3.weight')
    checkpoint.pop('classifier.3.bias')
    checkpoint.pop('classifier.6.weight')
    checkpoint.pop('classifier.6.bias')
    
    model.load_state_dict(checkpoint, strict=False)
    return model


class VGG4CAM(nn.Module):
    def __init__(self, args): # use args
        super(VGG4CAM, self).__init__()
        self.num_classes = args.num_classes
        
        self.vgg16 = vgg()

        self.features = self.vgg16.features
        # self.avgpool = self.vgg16.avgpool
        self.classifier = nn.Conv2d(512, self.num_classes, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        
        self.backbone = nn.ModuleList([self.features])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.features(x)
        # x = self.avgpool(x)
        x = self.classifier(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(-1, 20)

        return x
    
    def forward_cam(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.relu(x)

        return x

    def train(self, mode=True):
        for p in self.vgg16.features[0].parameters():
            p.requires_grad = False
        for p in self.vgg16.features[1].parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))

        

class VGGCAM(VGG4CAM):

    def __init__(self, args):
        super(VGGCAM, self).__init__(args)

    def forward(self, x):
        
        x = self.features(x)
        x = self.classifier(x)
        x = F.relu(x)

        x = x[0] + x[1].flip(-1)

        return x