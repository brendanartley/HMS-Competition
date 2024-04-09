import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivLossWithLogits(nn.KLDivLoss):
    def __init__(self, reduction="batchmean"):
        super().__init__(reduction=reduction)

    def forward(self, y, t):
        y = F.log_softmax(y, dim=1)
        loss = super().forward(y, t)
        return loss

class WeightedKLDivWithLogitsLoss(nn.Module):
    """
    Modified from: https://github.com/bdsp-core/IIIC-SPaRCNet/tree/main
    """
    def __init__(self, ):
        super().__init__()

    def forward(self, input, target, weight, label_smoothing=0.0):
        batch_size = input.size(0)
        log_prob = F.log_softmax(input, 1)
        
        # Optional: Smooth labels
        smoothed_target = (1 - label_smoothing) * target + label_smoothing / target.size(1)
        element_loss = F.kl_div(log_prob, smoothed_target, reduction='none')

        sample_loss = torch.sum(element_loss, dim=1)
        sample_weight = torch.sum(target * weight, dim=1)

        weighted_loss = sample_loss * sample_weight
        avg_loss = torch.sum(weighted_loss) / batch_size
        return avg_loss

if __name__ == "__main__":

    # Generate random logits and probabilities
    logits = torch.randn(10, 5)
    probabilities = F.softmax(torch.randn(10, 5), dim=1)
    weights = torch.ones(10, 5) # if all=1, then scores should match
    weights[0, :] = torch.zeros(5)

    # Loss objects
    kl_loss = KLDivLossWithLogits()
    weighted_kl_loss = WeightedKLDivWithLogitsLoss()

    # Calc loss
    kl_loss_value = kl_loss(logits, probabilities)
    weighted_kl_loss_value = weighted_kl_loss(logits, probabilities, weights)

    print("KLDiv:", kl_loss_value.item())
    print("KLDiv weighted:", weighted_kl_loss_value.item())