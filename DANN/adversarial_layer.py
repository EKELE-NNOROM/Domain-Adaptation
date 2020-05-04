import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

class ReversalLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, n_input=256, n_hidden=256):
        super(DomainDiscriminator, self).__init__()
        self.input_dim = n_input
        self.hidden_dim = n_hidden
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(self.bn(x))
        x = torch.sigmoid(x)
        return x
