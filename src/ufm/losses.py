"""Loss functions: weighted cross-entropy (dataset authors' baseline) and
focal + Dice (proposed, targets the rare flooded-urban class)."""
import segmentation_models_pytorch as smp
import torch
from torch import nn

from .bands import IGNORE


def enet_weights(class_counts, c=1.02):
    """ENet-style class weights 1 / ln(c + freq): strong but bounded rebalancing."""
    f = torch.tensor(class_counts, dtype=torch.float64)
    f = f / f.sum()
    return (1.0 / torch.log(c + f)).float()


class FocalDice(nn.Module):
    def __init__(self, gamma=2.0, dice_weight=1.0, dice_classes=(1, 2)):
        super().__init__()
        self.focal = smp.losses.FocalLoss("multiclass", gamma=gamma, ignore_index=IGNORE)
        self.dice = smp.losses.DiceLoss("multiclass", classes=list(dice_classes),
                                        from_logits=True, ignore_index=IGNORE)
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.focal(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(cfg, class_counts):
    l = cfg["loss"]
    if l["name"] == "wce":
        return nn.CrossEntropyLoss(weight=enet_weights(class_counts), ignore_index=IGNORE)
    if l["name"] == "focal_dice":
        return FocalDice(gamma=l.get("gamma", 2.0), dice_weight=l.get("dice_weight", 1.0))
    raise ValueError(l["name"])
