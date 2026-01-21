import copy
import torch


class EMA:
    """
    Exponential Moving Average (EMA) helper for the student model.

    We keep a separate, non-trainable copy of the model parameters and update it
    with an exponential moving average of the student's weights. This EMA model
    is then used as the teacher in CMT.
    """

    def __init__(self, model: torch.nn.Module, ema_decay: float):
        self.ema_decay = float(ema_decay)
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """
        Update EMA parameters from the current student model.
        """
        beta = self.ema_decay
        for p_ema, p in zip(self.ema_model.parameters(), model.parameters()):
            if p is None:
                continue
            p_ema.data.mul_(beta).add_(p.data, alpha=1.0 - beta)


