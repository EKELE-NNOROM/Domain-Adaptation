import torch
import torch.nn as nn

import adversarial_layer
from utils import DEVICE

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature = nn.Sequential(
                                nn.Conv2d(3, 64, kernel_size=5),
                                nn.BatchNorm2d(64),
                                nn.MaxPool2d(2),
                                nn.ReLU(True),
                                nn.Conv2d(64, 50, kernel_size=5),
                                nn.BatchNorm2d(50),
                                nn.Dropout2d(),
                                nn.MaxPool2d(2),
                                nn.ReLU(True)
        )

    def forward(self, x):
        return self.feature(x)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential(
                                              nn.Linear(50*4*4, 100),
                                              nn.BatchNorm1d(100),
                                              nn.ReLU(True),
                                              nn.Dropout2d(),
                                              nn.Linear(100, 100),
                                              nn.BatchNorm1d(100),
                                              nn.ReLU(True),
                                              nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.class_classifier(x)


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature = FeatureExtractor()
        self.classifier = Classifier()
        self.domain_classifier = adversarial_layer.DomainDiscriminator(n_input=50*4*4, n_hidden=100)

    def forward(self, input, alpha=1, source=True):
        input = input.expand(len(input), 3, 28, 28)
        feature = self.feature(input)
        feature = feature.view(-1, 50*4*4)
        class_output = self.classifier(feature)
        domain_output = self.get_adversarial_result(feature, source, alpha)
        return class_output, domain_output

    def get_adversarial_result(self, x, source=True, alpha=1):
        loss_fn = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(x)).long().to(DEVICE)
        else:
            domain_label = torch.zeros(len(x)).long().to(DEVICE)
        x = adversarial_layer.ReversalLayerF.apply(x, alpha)
        domain_pred = self.domain_classifier(x)
        loss_adv = loss_fn(domain_pred, domain_label.float())
        return loss_adv
