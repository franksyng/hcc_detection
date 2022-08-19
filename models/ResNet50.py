import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch.nn as nn
import torch.nn.functional as F


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.feature_extractor = create_feature_extractor(resnet50, return_nodes=['layer4'])
        self.batch_norm = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.time_classifier = nn.Sequential(
            nn.Linear(2048, 3),
        )
        self.gene_classifier = nn.Sequential(
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        x = self.feature_extractor(x)['layer4']
        x = F.interpolate(x, (5, 5), mode='bilinear', align_corners=False)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        time = self.time_classifier(self.flatten(x))
        gene = self.gene_classifier(self.flatten(x))
        return gene, time

