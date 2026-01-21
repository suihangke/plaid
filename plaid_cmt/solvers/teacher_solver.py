import torch
from torch import nn


class TeacherSolver(nn.Module):
    """
    Multi-step teacher solver used in CMT to generate a high-quality mapping
    from x_t at time t to an approximate x_0 in embedding space.

    This implementation follows the exact sampling logic from sample.py:
    - Uses the same time discretization (linspace from t to 0)
    - Uses the same diffusion step update (DDIM or VDM-style)
    - Maintains self-conditioning throughout the trajectory
    """

    def __init__(
        self,
        adapter,
        noise_schedule,
        gamma_bounds,
        n_steps: int,
        cache=None,
        ddim_sampler: bool = False,
        score_temp: float = 0.9,
    ):
        """
        Args:
            adapter: PlaidDiffusionAdapter using the EMA teacher weights.
            noise_schedule: lib.models.NoiseSchedule instance.
            gamma_bounds: lib.models.GammaBounds instance.
            n_steps: Number of solver steps from t to 0.
            cache: Optional cache object with get/put API for storing teacher
                outputs keyed by (sample_id, t).
            ddim_sampler: If True, use DDIM sampling; otherwise use VDM-style sampling.
            score_temp: Temperature scaling for the score function (from sample.py).
        """
        super().__init__()
        self.adapter = adapter
        self.noise_schedule = noise_schedule
        self.gamma_bounds = gamma_bounds
        self.n_steps = n_steps
        self.cache = cache
        self.ddim_sampler = ddim_sampler
        self.score_temp = score_temp

    @torch.no_grad()
    def forward(self, x_t, t_scalar: float, input_ids=None, cache_key=None):
        """
        Run the multi-step teacher trajectory from time t to 0, following
        the exact logic from sample.py.

        Args:
            x_t: Noisy state at time t, shape [B, L, D].
            t_scalar: Python float in [0, 1] representing the starting time.
            input_ids: Optional conditioning tokens, shape [B, L] (currently unused).
            cache_key: Optional key used for caching teacher outputs.

        Returns:
            x0_teacher: [B, L, D] approximate clean embedding state.
        """
        if self.cache is not None and cache_key is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        device = x_t.device
        gamma_0, gamma_1 = self.gamma_bounds()

        # Construct time grid from t down to 0, matching sample.py logic.
        t_start = float(t_scalar)
        t_steps = torch.linspace(t_start, 0.0, self.n_steps + 1, device=device, dtype=torch.float64)

        z = x_t.clone().double()
        x_selfcond = torch.zeros_like(z).float()

        # Iterate through time steps from t to 0.
        for i, t in enumerate(t_steps[:-1]):  # Exclude the final 0.0 step
            t = t[None]  # [1]
            s = t_steps[i + 1][None]  # Next time step [1]

            # Map t and s to gamma using the learned noise schedule.
            gamma_tilde_t = self.noise_schedule(t).double()  # [1]
            gamma_tilde_s = self.noise_schedule(s).double()  # [1]
            gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_tilde_t  # [1]
            gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_tilde_s  # [1]

            # Compute alpha and sigma at times t and s.
            alpha_squared_t = torch.sigmoid(-gamma_t)
            alpha_squared_s = torch.sigmoid(-gamma_s)
            alpha_t = alpha_squared_t.sqrt()
            alpha_s = alpha_squared_s.sqrt()
            sigma_squared_t = torch.sigmoid(gamma_t)
            sigma_squared_s = torch.sigmoid(gamma_s)
            sigma_t = sigma_squared_t.sqrt()
            sigma_s = sigma_squared_s.sqrt()

            # Single diffusion step: predict x_reconst (approximate x0).
            _, x_reconst = self.adapter(
                z_t=z.to(torch.float32, copy=True),
                gamma_t=gamma_t.float(),
                input_ids=input_ids,
                x_selfcond=x_selfcond,
                selfcond_mask=None,
            )
            x_selfcond = x_reconst.clone().detach()
            x_reconst = x_reconst.double()

            # Compute epsilon prediction and update x_reconst (matching sample.py).
            epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
            epsilon_pred /= self.score_temp
            x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t

            # Update z to the next time step.
            if s.item() > 0:
                if self.ddim_sampler:
                    # DDIM update: deterministic
                    z = (alpha_s * x_reconst) + (sigma_s * epsilon_pred)
                else:
                    # VDM-style update: includes stochastic term
                    c = -torch.expm1(gamma_s - gamma_t)
                    z = z * (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                    z = z + c * (alpha_squared_s.sqrt() * x_reconst)
                    z = z + (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)
            else:
                # At the final step (s=0), just use x_reconst as the final output.
                z = x_reconst

        # Final forward pass to get the clean embedding state.
        # At this point z should be close to x0, but we do one more forward
        # to ensure consistency (matching sample.py's final step).
        gamma_tilde_final = self.noise_schedule(torch.tensor([0.0], device=device)).double()
        gamma_final = gamma_0 + (gamma_1 - gamma_0) * gamma_tilde_final
        _, x_reconst_final = self.adapter(
            z_t=z.float(),
            gamma_t=gamma_final.float(),
            input_ids=input_ids,
            x_selfcond=x_selfcond,
            selfcond_mask=None,
        )
        x0_teacher = x_reconst_final[:, :, : self.adapter.embed_dim]

        if self.cache is not None and cache_key is not None:
            self.cache.put(cache_key, x0_teacher)
        return x0_teacher


