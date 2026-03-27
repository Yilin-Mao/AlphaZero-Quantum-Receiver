import math
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np


# ============================================================
# Observation builders
# ============================================================
def build_obs_standard(
    p: np.ndarray,
    t: int,
    T: int,
    a_rem: float,
    alpha: float,
    r_t: float,
) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    obs = np.concatenate([
        np.log(p + 1e-8),
        np.array([
            float(t) / float(T),
            float(a_rem) / float(alpha + 1e-12),
            math.sqrt(max(0.0, float(r_t))),
            float(alpha),
        ], dtype=np.float64),
    ])
    return obs.astype(np.float32)


def build_obs_chirped(
    p: np.ndarray,
    t: int,
    T: int,
    a_rem: float,
    alpha: float,
    r_t: float,
    delta_t: float,
) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    obs = np.concatenate([
        np.log(p + 1e-8),
        np.array([
            float(t) / float(T),
            float(a_rem) / float(alpha + 1e-12),
            math.sqrt(max(0.0, float(r_t))),
            float(alpha),
            math.cos(delta_t),
            math.sin(delta_t),
        ], dtype=np.float64),
    ])
    return obs.astype(np.float32)


# ============================================================
# Base helpers
# ============================================================
def default_r_per_step(T: int):
    return [1.0 / i for i in range(T, 0, -1)]


class BaseKPSKEnv:
    """
    Shared interface for all KPSK environments.
    Child classes should keep:
      - self.K, self.alpha, self.eta, self.p_dark, self.T
      - self.p, self.a_rem, self.t, self.m
      - methods:
            reset_episode()
            get_obs()
            normalized_action_to_beta()
            step(beta)
            final_reward()
            get_obs_dim()
            get_symbol_fields(t, scale)
    """

    def _click_prob(self, alpha_eff: complex) -> float:
        mu = abs(alpha_eff) ** 2
        p_no = math.exp(-self.eta * mu) * (1.0 - self.p_dark)
        return 1.0 - p_no

    def normalized_action_to_beta(self, a: np.ndarray) -> complex:
        r = float(self.r_per_step[self.t])
        scale = math.sqrt(r) * float(self.a_rem)
        return complex(float(a[0]), float(a[1])) * scale

    def final_reward(self) -> float:
        if self.guess is None:
            self.guess = int(np.argmax(self.p))
            self.guess_correct = (self.guess == self.m)
        return 1.0 if self.guess_correct else 0.0

    def get_obs_dim(self) -> int:
        raise NotImplementedError

    def get_symbol_fields(self, t: int, scale: float) -> np.ndarray:
        raise NotImplementedError


# ============================================================
# Standard environment
# ============================================================
class KPSKEnv(BaseKPSKEnv):
    def __init__(
        self,
        K=4,
        alpha=1.0,
        eta=1.0,
        p_dark=0.0,
        T=6,
        r_per_step=None,
        seed=0,
    ):
        self.K = int(K)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.p_dark = float(p_dark)
        self.T = int(T)
        self.r_per_step = default_r_per_step(self.T) if r_per_step is None else list(r_per_step)
        self.rng = random.Random(seed)

        self.phis = np.array([2.0 * math.pi * m / self.K for m in range(self.K)], dtype=np.float64)
        self.alphabet = np.array([
            self.alpha * complex(math.cos(phi), math.sin(phi)) for phi in self.phis
        ], dtype=np.complex128)

        self.reset_episode()

    def get_obs_dim(self) -> int:
        return self.K + 4

    def reset_episode(self):
        self.m = self.rng.randrange(self.K)
        self.p = np.ones(self.K, dtype=np.float64) / self.K
        self.a_rem = self.alpha
        self.t = 0
        self.guess = None
        self.guess_correct = None
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        return build_obs_standard(
            p=self.p,
            t=self.t,
            T=self.T,
            a_rem=self.a_rem,
            alpha=self.alpha,
            r_t=self.r_per_step[self.t] if self.t < self.T else 0.0,
        )

    def get_symbol_fields(self, t: int, scale: float) -> np.ndarray:
        return self.alphabet * scale

    def step(self, beta: complex):
        r = float(self.r_per_step[self.t])
        sqrt_r = math.sqrt(r)
        scale = self.a_rem / self.alpha if self.alpha > 0 else 1.0

        symbol_fields = self.get_symbol_fields(self.t, scale=scale)

        a_true = sqrt_r * symbol_fields[self.m] - beta
        p1_true = self._click_prob(a_true)
        z = 1 if self.rng.random() < p1_true else 0

        L = np.empty(self.K, dtype=np.float64)
        for j in range(self.K):
            a_j = sqrt_r * symbol_fields[j] - beta
            p1_j = self._click_prob(a_j)
            L[j] = p1_j if z == 1 else (1.0 - p1_j)

        self.p = self.p * L
        self.p = self.p / (self.p.sum() + 1e-12)

        self.a_rem = math.sqrt(max(0.0, 1.0 - r)) * self.a_rem
        self.t += 1
        done = self.t >= self.T

        if done:
            self.guess = int(np.argmax(self.p))
            self.guess_correct = (self.guess == self.m)

        return (self.get_obs() if not done else None), z, done


# ============================================================
# Chirped / oscillatory hard environment
# ============================================================
class ChirpedKPSKEnv(BaseKPSKEnv):
    def __init__(
        self,
        K=4,
        alpha=1.0,
        eta=1.0,
        p_dark=0.0,
        T=12,
        seed=0,
        chirp_amp_1=0.42,
        chirp_freq_1=3.0,
        chirp_amp_2=0.18,
        chirp_freq_2=7.0,
        chirp_phase_2=0.3,
        energy_omega=3.5,
        energy_floor=0.18,
        use_oscillatory_schedule=True,
    ):
        self.K = int(K)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.p_dark = float(p_dark)
        self.T = int(T)
        self.rng = random.Random(seed)

        self.chirp_amp_1 = float(chirp_amp_1)
        self.chirp_freq_1 = float(chirp_freq_1)
        self.chirp_amp_2 = float(chirp_amp_2)
        self.chirp_freq_2 = float(chirp_freq_2)
        self.chirp_phase_2 = float(chirp_phase_2)

        self.energy_omega = float(energy_omega)
        self.energy_floor = float(energy_floor)
        self.use_oscillatory_schedule = bool(use_oscillatory_schedule)

        self.base_phis = np.array([2.0 * math.pi * m / self.K for m in range(self.K)], dtype=np.float64)

        # keep phis as standard PSK phases, so heuristic baseline still uses standard coarse phase grid
        self.phis = self.base_phis.copy()

        self.reset_episode()

    def get_obs_dim(self) -> int:
        return self.K + 6

    def phase_warp(self, t: int) -> float:
        x = (t + 1) / max(self.T, 1)
        return float(
            self.chirp_amp_1 * math.sin(2.0 * math.pi * self.chirp_freq_1 * x) +
            self.chirp_amp_2 * math.sin(2.0 * math.pi * self.chirp_freq_2 * x + self.chirp_phase_2)
        )

    def make_r_per_step(self):
        if not self.use_oscillatory_schedule:
            return default_r_per_step(self.T)

        raw = []
        for t in range(self.T):
            x = (t + 1) / max(self.T, 1)
            val = self.energy_floor + (1.0 - self.energy_floor) * (math.sin(math.pi * self.energy_omega * x) ** 2)
            raw.append(float(val))

        suffix = np.cumsum(raw[::-1])[::-1]
        return [raw[t] / max(suffix[t], 1e-12) for t in range(self.T)]

    def current_alphabet(self, t: int, alpha_scale: float = 1.0):
        delta = self.phase_warp(t)
        phis_t = self.base_phis + delta
        return np.array([
            (self.alpha * alpha_scale) * complex(math.cos(phi), math.sin(phi))
            for phi in phis_t
        ], dtype=np.complex128)

    def reset_episode(self):
        self.m = self.rng.randrange(self.K)
        self.p = np.ones(self.K, dtype=np.float64) / self.K
        self.a_rem = self.alpha
        self.t = 0
        self.guess = None
        self.guess_correct = None
        self.r_per_step = self.make_r_per_step()
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        delta_t = self.phase_warp(self.t) if self.t < self.T else 0.0
        r_t = self.r_per_step[self.t] if self.t < self.T else 0.0
        return build_obs_chirped(
            p=self.p,
            t=self.t,
            T=self.T,
            a_rem=self.a_rem,
            alpha=self.alpha,
            r_t=r_t,
            delta_t=delta_t,
        )

    def get_symbol_fields(self, t: int, scale: float) -> np.ndarray:
        return self.current_alphabet(t, alpha_scale=scale)

    def step(self, beta: complex):
        r = float(self.r_per_step[self.t])
        sqrt_r = math.sqrt(r)
        scale = self.a_rem / self.alpha if self.alpha > 0 else 1.0

        symbol_fields = self.get_symbol_fields(self.t, scale=scale)

        a_true = sqrt_r * symbol_fields[self.m] - beta
        p1_true = self._click_prob(a_true)
        z = 1 if self.rng.random() < p1_true else 0

        L = np.empty(self.K, dtype=np.float64)
        for j in range(self.K):
            a_j = sqrt_r * symbol_fields[j] - beta
            p1_j = self._click_prob(a_j)
            L[j] = p1_j if z == 1 else (1.0 - p1_j)

        self.p = self.p * L
        s = float(self.p.sum())
        self.p = self.p / (s + 1e-12)

        self.a_rem = math.sqrt(max(0.0, 1.0 - r)) * self.a_rem
        self.t += 1
        done = self.t >= self.T

        if done:
            self.guess = int(np.argmax(self.p))
            self.guess_correct = (self.guess == self.m)

        return (self.get_obs() if not done else None), z, done


# ============================================================
# Factory
# ============================================================
ENV_REGISTRY = {
    "standard_kpsk": KPSKEnv,
    "chirped_kpsk": ChirpedKPSKEnv,
}


def make_env(env_name: str = "standard_kpsk", **kwargs):
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown env_name={env_name}. Available: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[env_name](**kwargs)