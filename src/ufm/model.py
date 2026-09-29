"""U-Net (ResNet-34 encoder) wrapped with the input builder, plus MC-dropout."""
import segmentation_models_pytorch as smp
import torch
from torch import nn

from .features import InputBuilder


class FloodNet(nn.Module):
    def __init__(self, channels, mean, std, encoder="resnet34", encoder_weights="imagenet",
                 dropout=0.2, n_classes=3):
        super().__init__()
        self.inputs = InputBuilder(channels, mean, std)
        self.net = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                            in_channels=len(channels), classes=n_classes)
        if dropout > 0:
            # dropout on the decoder output: regularises training and, kept active
            # at test time, gives Monte-Carlo samples for uncertainty
            self.net.segmentation_head = nn.Sequential(nn.Dropout2d(dropout), self.net.segmentation_head)

    def forward(self, raw, aux, coh_offset):
        return self.net(self.inputs(raw, aux, coh_offset))


def build_model(cfg, mean, std, pretrained=True):
    m = cfg["model"]
    return FloodNet(cfg["channels"], mean, std, encoder=m.get("encoder", "resnet34"),
                    encoder_weights=m.get("encoder_weights", "imagenet") if pretrained else None,
                    dropout=m.get("dropout", 0.2))


def enable_mc_dropout(model):
    model.eval()
    for mod in model.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout2d)):
            mod.train()


@torch.no_grad()
def predict(model, raw, aux, off, mc_samples=0):
    """Returns (probabilities (B,C,H,W), uncertainty (B,H,W) or None).

    With mc_samples > 0: mean softmax over MC-dropout passes, and predictive
    entropy (in nats) as the uncertainty map.
    """
    if mc_samples <= 0:
        model.eval()
        return torch.softmax(model(raw, aux, off).float(), 1), None
    enable_mc_dropout(model)
    p = torch.stack([torch.softmax(model(raw, aux, off).float(), 1) for _ in range(mc_samples)]).mean(0)
    model.eval()
    entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(1)
    return p, entropy
