import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision

nclasses = 20 

class SimpleNet(nn.Module):
    """A simple self defined network, input image size should be 112
    """
    def __init__(self, dropout=0.3):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.drop = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 7 * 7, 500)
        self.fc2 = nn.Linear(500, nclasses)

    def forward(self, x, dropout=0.3):
        x = F.relu(F.max_pool2d(self.bn1(self.drop(self.conv1(x))), 2))
        x = F.relu(F.max_pool2d(self.bn2(self.drop(self.conv2(x))), 2))
        x = F.relu(F.max_pool2d(self.bn3(self.drop(self.conv3(x))), 2))
        x = F.relu(F.max_pool2d(self.bn4(self.drop(self.conv4(x))), 2))
        x = x.view(-1, 256 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def TransferLearning(netname, num_classes=20):
    # size of input image should be 224
    if netname == "resnet":
        model = models.resnet18(pretrained=True)
        # ---------freeze all layers------------#
        # for param in model.parameters():
        #     param.requires_grad = False

        # ------only change params in "layer1"-------- #
        # ct = []
        # for name, child in model_ft.named_children():
        #     if "layer1" in ct:
        #         for params in child.parameters():
        #             params.requires_grad = True
        #     ct.append(name)

        # --------freeze first 7 layers-------- #
        ct = 0
        for name, child in model.named_children():
            ct += 1
            if ct < 7:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

        return model

    if netname == "vgg":
        model = torchvision.models.vgg16(pretrained=True)

        model.features.requires_grad = False
        model.classifier.requires_grad = True

        model.classifier[6].out_features = num_classes

        return model

    if netname == "densenet":
        model = torchvision.models.densenet121(pretrained=True)
        
        ct = []
        for name, child in model.features.named_children():
            if "denseblock1" in ct:
                for params in child.parameters():
                    params.requires_grad = True
            ct.append(name)

        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

        return model

    if netname == "squeezenet":
        model = torchvision.models.squeezenet1_0(pretrained=True)
        
        ct = []
        for name, child in model.features.named_children():
            if "3" in ct:
                for params in child.parameters():
                    params.requires_grad = True
            ct.append(name)

        in_ftrs = model.classifier[1].in_channels
        out_ftrs = model.classifier[1].out_channels
        features = list(model.classifier.children())
        features[1] = nn.Conv2d(in_ftrs, num_classes, kernel_size=(2, 2), stride=(1, 1))
        features[3] = nn.AvgPool2d(12, stride=1)
        model.classifier = nn.Sequential(*features)
        model.num_classes = num_classes

        return model

    if netname == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        
        ct = []
        for name, child in model.features.named_children():
            if "5" in ct:
                for params in child.parameters():
                    params.requires_grad = True
            ct.append(name)

        num_ftrs = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]
        features.extend([nn.Linear(num_ftrs, num_classes)])
        model.classifier = nn.Sequential(*features)

        return model

    # for inception v3 model, size of image should be 299
    if netname == "inceptionv3":
        model = torchvision.models.inception_v3(pretrained=True)
        
        ct = []
        for name, child in model.named_children():
            if "Conv2d_4a_3x3" in ct:
                for params in child.parameters():
                    params.requires_grad = True
            ct.append(name)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        return model

    print('Model should be one of ["resnet", "vgg", "densenet", "squeezenet", "alexnet", "inceptionv3"]')
    return None

