# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：ot_module.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/7/29 11:45 
"""
import torch
from torch import nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    """Leaves the input unchanged during forward propagation
    and reverses the gradient by multiplying it by a
    negative scalar during the backpropagation.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        """XXX add docstring here."""
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """XXX add docstring here."""
        output = grad_output.neg() * ctx.alpha
        return output, None
class ot_classifier(nn.Module):
    """Classifier Architecture from DANN paper [15]_.

       Parameters
       ----------
       num_features : int
           Size of the input, e.g size of the last layer of
           the feature extractor
       n_classes : int, default=1
           Number of classes

       References
       ----------
       .. [15]  Yaroslav Ganin et. al. Domain-Adversarial Training
               of Neural Networks  In Journal of Machine Learning
               Research, 2016.
       """

    def __init__(self, num_features, n_classes=1, alpha=1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, n_classes),
            nn.Softmax(),
        )
        self.alpha = alpha

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        alpha: float
            Parameter for the reverse layer.
        """
        reverse_x = GradientReversalLayer.apply(x, self.alpha)
        temp=self.classifier(reverse_x)

        return temp
