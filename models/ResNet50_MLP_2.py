import torchvision.models as models
import torch.nn as nn


class ResNet50_MLP_2(nn.Module):
    def __init__(self):
        super(ResNet50_MLP_2, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        prev_dim = self.resnet.fc.weight.shape[1]
        self.resnet.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # 1st layer
            nn.Linear(in_features=2048, out_features=2048),
            nn.BatchNorm1d(2048, affine=False))  # output layer
        self.resnet.fc[3].bias.requires_grad = False
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

