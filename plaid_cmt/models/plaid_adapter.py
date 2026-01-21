import torch
from torch import nn


class PlaidDiffusionAdapter(nn.Module):
    """
    Thin adapter around the Plaid diffusion LM modules to provide a clean
    teacher/student interface for CMT:
      - encode(input_ids) -> x0 in embedding space
      - forward(z_t, gamma_t, input_ids, x_selfcond) -> (logits, x_reconst)
      - decode_from_embeddings(x0) -> logits over vocab (simple tied decoder)

    We assume the following Plaid modules dict:
      {
        'noise_schedule': lib.models.NoiseSchedule(),
        'gamma_bounds':  lib.models.GammaBounds(gamma_0, gamma_1),
        'embedding_matrix': lib.models.EmbeddingMatrix(vocab_size, embed_dim),
        'model': lib.models.DiffusionModel(dim, embed_dim, n_blocks, n_heads, vocab_size),
      }
    """

    def __init__(self, modules, vocab_size, embed_dim, device):
        super().__init__()
        # Register modules as submodules so they are included in parameters()
        self.noise_schedule = modules["noise_schedule"]
        self.gamma_bounds = modules["gamma_bounds"]
        self.embedding_matrix = modules["embedding_matrix"]
        self.model = modules["model"]
        # Keep reference to dict for backward compatibility
        self.modules = modules
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.device = device

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Map discrete token ids to continuous embeddings using the Plaid
        embedding_matrix module.

        Args:
            input_ids: Long tensor of shape [B, L].
        Returns:
            x0: Float tensor of shape [B, L, D] in embedding space.
        """
        emb_matrix = self.modules["embedding_matrix"]().to(self.device)  # [V, D]
        x0 = torch.nn.functional.embedding(input_ids, emb_matrix)
        return x0

    def forward(
        self,
        z_t: torch.Tensor,
        gamma_t: torch.Tensor,
        input_ids: torch.Tensor,
        x_selfcond: torch.Tensor,
        selfcond_mask: torch.Tensor | None = None,
    ):
        """
        Single diffusion forward step using the Plaid DiffusionModel.

        Args:
            z_t: Noisy state at time t, shape [B, L, D].
            gamma_t: Gamma value at time t, shape [B] (one per batch item).
            input_ids: Condition tokens, shape [B, L] (currently unused but
                kept for future conditional extensions).
            x_selfcond: Self-conditioning signal, same shape as z_t.
            selfcond_mask: Optional mask for self-conditioning, shape [B].
        Returns:
            logits: [B, L, V]
            x_reconst: [B, L, D] denoised reconstruction in embedding space.
        """
        logits, x_reconst = self.modules["model"](
            z=z_t.to(torch.float32, copy=True),
            gamma=gamma_t.float(),
            embedding_matrix=self.modules["embedding_matrix"](),
            bias_scale=1.0,
            x_selfcond=x_selfcond,
            selfcond_mask=selfcond_mask,
        )
        # x_reconst already lives in embedding space with last dim == embed_dim.
        x_reconst = x_reconst[:, :, : self.embed_dim]
        return logits, x_reconst

    @torch.no_grad()
    def decode_from_embeddings(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Simple tied-embedding decoder from embedding space back to logits.
        This is only used for inspection / sampling, not for the core CMT
        loss, which stays in embedding space.

        Args:
            x0: [B, L, D] embeddings.
        Returns:
            logits: [B, L, V]
        """
        emb_matrix = self.modules["embedding_matrix"]().to(self.device)  # [V, D]
        logits = torch.einsum("bld,vd->blv", x0, emb_matrix)
        return logits


