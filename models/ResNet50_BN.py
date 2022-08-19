import torchvision.models as models
import torch.nn as nn


class ResNet50_BN(nn.Module):
    def __init__(self):
        super(ResNet50_BN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        prev_dim = self.resnet.fc.weight.shape[1]
        self.resnet.fc = nn.Sequential(
            nn.Linear(prev_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),  # 1st layer
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),  # 2nd layer
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),  # 3rd layer
            nn.Linear(128, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),  # 4th layer
            nn.Linear(in_features=512, out_features=2048),
            nn.BatchNorm1d(2048, affine=False))  # output layer
        self.resnet.fc[12].bias.requires_grad = False
        self.relu = nn.ReLU()
        self.gene_classifier = nn.Sequential(
            nn.Linear(2048, 2)
        )
        self.time_classifier = nn.Sequential(
            nn.Linear(2048, 3)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.relu(x)
        gene = self.gene_classifier(x)
        time = self.time_classifier(x)
        return gene, time

