import math
import numpy as np


def helstrom_kpsk_pure_closed_form(alpha: float, K: int) -> float:
    """
    Helstrom/YKL optimal success probability for pure-state KPSK coherent states
    with equal priors, using the SRM/DFT closed form.

    Parameters
    ----------
    alpha : float
        Real amplitude magnitude |alpha|.
    K : int
        Number of PSK phases.

    Returns
    -------
    float
        Optimal success probability P_succ.
    """
    Ns = float(alpha) ** 2

    s = np.arange(K, dtype=np.float64)
    theta = 2.0 * np.pi * s / K
    g = np.exp(
        -Ns * (1.0 - np.cos(theta)) +
        1j * (Ns * np.sin(theta))
    )

    lam = np.fft.fft(g)
    lam = np.real(lam)
    lam = np.clip(lam, 0.0, None)

    Pc = (np.sum(np.sqrt(lam)) ** 2) / (K ** 2)
    return float(Pc)


class DolinarLikePolicy:
    """
    Dolinar-like one-step lookahead policy for KPSK/BPSK.

    Two-stage search:
      1) coarse grid over beta = rho * |a_rem| * exp(i phi)
      2) local refinement around the coarse best beta

    This version is compatible with both:
      - standard KPSKEnv (static alphabet)
      - ChirpedKPSKEnv (time-varying alphabet)

    The key change is that the effective symbol fields are queried from the env
    at the CURRENT STEP whenever possible.
    """

    def __init__(
        self,
        env,
        beta_grid_coarse=None,
        phase_grid_coarse=None,
        refine_rho_span=0.15,
        refine_phi_span=math.pi / 18,
        beta_grid_refine=None,
        phase_grid_refine=None,
    ):
        self.env = env

        # ---------- stage 1: coarse ----------
        self.beta_grid_coarse = (
            np.asarray(beta_grid_coarse, dtype=np.float64)
            if beta_grid_coarse is not None
            else np.linspace(0.0, 1.0, 15)
        )

        if phase_grid_coarse is not None:
            self.phase_grid_coarse = np.asarray(phase_grid_coarse, dtype=np.float64)
        else:
            if hasattr(env, "phis"):
                # For ChirpedKPSKEnv, env.phis is intentionally kept as the standard PSK phases.
                # This preserves the baseline's coarse-grid bias and is useful for comparison.
                self.phase_grid_coarse = np.asarray(env.phis, dtype=np.float64)
            else:
                self.phase_grid_coarse = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)

        # ---------- stage 2: local refine ----------
        self.refine_rho_span = float(refine_rho_span)
        self.refine_phi_span = float(refine_phi_span)

        self.beta_grid_refine = (
            np.asarray(beta_grid_refine, dtype=np.float64)
            if beta_grid_refine is not None
            else np.linspace(-1.0, 1.0, 11)
        )
        self.phase_grid_refine = (
            np.asarray(phase_grid_refine, dtype=np.float64)
            if phase_grid_refine is not None
            else np.linspace(-1.0, 1.0, 11)
        )

    # --------------------------------------------------------
    # helpers
    # --------------------------------------------------------
    def _get_posterior(self) -> np.ndarray:
        if hasattr(self.env, "p"):
            p = np.asarray(self.env.p, dtype=np.float64)
            return p / (p.sum() + 1e-12)

        # fallback for older BPSK-style envs
        if hasattr(self.env, "p_plus"):
            p_plus = float(self.env.p_plus)
            return np.array([p_plus, 1.0 - p_plus], dtype=np.float64)

        raise AttributeError("Environment does not expose posterior as env.p or env.p_plus.")

    def _get_symbol_fields(self, t: int, scale: float) -> np.ndarray:
        """
        Return the CURRENT STEP symbol fields already scaled by remaining amplitude.

        Preferred path:
            env.get_symbol_fields(t, scale)

        Fallback:
            use env.alphabet * scale

        Final fallback:
            construct from env.phis
        """
        if hasattr(self.env, "get_symbol_fields"):
            fields = self.env.get_symbol_fields(t, scale=scale)
            return np.asarray(fields, dtype=np.complex128)

        if hasattr(self.env, "alphabet"):
            return np.asarray(self.env.alphabet, dtype=np.complex128) * scale

        if hasattr(self.env, "phis") and hasattr(self.env, "alpha"):
            alpha = float(self.env.alpha)
            phis = np.asarray(self.env.phis, dtype=np.float64)
            return np.array(
                [
                    (alpha * scale) * complex(math.cos(phi), math.sin(phi))
                    for phi in phis
                ],
                dtype=np.complex128,
            )

        # very old BPSK fallback
        if hasattr(self.env, "a_rem"):
            a = float(self.env.a_rem)
            return np.array([complex(+a, 0.0), complex(-a, 0.0)], dtype=np.complex128)

        raise AttributeError("Cannot infer symbol fields from environment.")

    def _exp_err_for_beta(self, p: np.ndarray, a0: np.ndarray, beta: complex) -> float:
        """
        One-step expected MAP error after on/off outcome, given:
          p   : posterior over hypotheses, shape (K,)
          a0  : undisplaced effective fields at current step, shape (K,)
          beta: complex displacement
        """
        K = len(p)
        p1 = np.zeros(K, dtype=np.float64)

        for m in range(K):
            a_eff = a0[m] - beta
            p1[m] = float(self.env._click_prob(a_eff))

        # click
        num1 = p * p1
        post1 = num1 / (num1.sum() + 1e-12)
        err1 = 1.0 - float(np.max(post1))

        # no click
        num0 = p * (1.0 - p1)
        post0 = num0 / (num0.sum() + 1e-12)
        err0 = 1.0 - float(np.max(post0))

        p_z1 = float(np.dot(p, p1))
        return p_z1 * err1 + (1.0 - p_z1) * err0

    # --------------------------------------------------------
    # main policy
    # --------------------------------------------------------
    def act(self) -> complex:
        p = self._get_posterior()
        K = len(p)

        r = float(self.env.r_per_step[self.env.t])
        sqrt_r = math.sqrt(r)

        a_rem = float(self.env.a_rem)
        alpha = float(self.env.alpha)
        scale = a_rem / alpha if alpha > 0 else 1.0

        symbol_fields = self._get_symbol_fields(t=self.env.t, scale=scale)

        # undisplaced effective fields per hypothesis at current step
        a0 = np.array(
            [sqrt_r * symbol_fields[m] for m in range(K)],
            dtype=np.complex128,
        )

        # -------------------------
        # stage 1: coarse search
        # -------------------------
        best_beta = 0.0 + 0.0j
        best_exp_err = 1e18
        abs_a = abs(a_rem)

        for rho in self.beta_grid_coarse:
            rho = float(rho)
            for phi in self.phase_grid_coarse:
                phi = float(phi)
                beta = (rho * abs_a) * complex(math.cos(phi), math.sin(phi))
                exp_err = self._exp_err_for_beta(p, a0, beta)
                if exp_err < best_exp_err:
                    best_exp_err = exp_err
                    best_beta = beta

        # -------------------------
        # stage 2: local refine
        # -------------------------
        best_rho = abs(best_beta) / (abs_a + 1e-12)
        best_phi = float(np.angle(best_beta))

        for dr_unit in self.beta_grid_refine:
            rho = best_rho + float(dr_unit) * self.refine_rho_span
            if rho < 0.0:
                continue

            for dphi_unit in self.phase_grid_refine:
                phi = best_phi + float(dphi_unit) * self.refine_phi_span
                phi = (phi + 2.0 * math.pi) % (2.0 * math.pi)

                beta = (rho * abs_a) * complex(math.cos(phi), math.sin(phi))
                exp_err = self._exp_err_for_beta(p, a0, beta)
                if exp_err < best_exp_err:
                    best_exp_err = exp_err
                    best_beta = beta

        return best_beta