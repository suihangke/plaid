import torch
from torch import nn


class ConsistencyLoss(nn.Module):
    """
    Basic consistency loss used in CMT:

        L_CMT = E[ || stopgrad(x0_teacher) - x0_student ||^2 ]

    where x0_teacher is the teacher's multi-step output and x0_student is
    the student's single-step prediction, both in embedding space.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(self, x0_teacher: torch.Tensor, x0_student: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0_teacher: Teacher output in embedding space, shape [B, L, D].
            x0_student: Student output in embedding space, shape [B, L, D].
        Returns:
            Scalar loss tensor.
        """
        target = x0_teacher.detach()
        loss = torch.mean((x0_student - target) ** 2)
        return self.weight * loss


