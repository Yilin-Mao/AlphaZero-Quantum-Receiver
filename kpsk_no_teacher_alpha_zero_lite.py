import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1) Environment: KPSK coherent-state discrimination with
#    displacement + on/off detection + Bayes update
# ============================================================
class KPSKEnv:
    def __init__(self, K=4, alpha=1.0, eta=1.0, p_dark=0.0, T=6,
                 r_per_step=None, seed=0):
        self.K = int(K)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.p_dark = float(p_dark)
        self.T = int(T)
        # equal remaining-energy usage schedule by default
        # r_t = 1/(remaining steps)
        self.r_per_step = [1.0 / i for i in range(self.T, 0, -1)] if r_per_step is None else list(r_per_step)
        self.rng = random.Random(seed)

        self.phis = np.array([2.0 * math.pi * m / self.K for m in range(self.K)], dtype=np.float64)
        self.alphabet = np.array([
            self.alpha * complex(math.cos(phi), math.sin(phi)) for phi in self.phis
        ], dtype=np.complex128)

        self.reset_episode()

    def reset_episode(self):
        self.m = self.rng.randrange(self.K)  # true symbol index
        self.p = np.ones(self.K, dtype=np.float64) / self.K
        self.a_rem = self.alpha
        self.t = 0
        self.guess = None
        self.guess_correct = None
        return self.get_obs()

    def _click_prob(self, alpha_eff: complex) -> float:
        mu = abs(alpha_eff) ** 2
        p_no = math.exp(-self.eta * mu) * (1.0 - self.p_dark)
        return 1.0 - p_no

    def get_obs(self) -> np.ndarray:
        return build_obs(
            p=self.p,
            t=self.t,
            T=self.T,
            a_rem=self.a_rem,
            alpha=self.alpha,
            r_t=self.r_per_step[self.t] if self.t < self.T else 0.0,
        )

    def normalized_action_to_beta(self, a: np.ndarray) -> complex:
        # a is normalized in [-1,1]^2
        r = float(self.r_per_step[self.t])
        scale = math.sqrt(r) * float(self.a_rem)
        return complex(float(a[0]), float(a[1])) * scale

    def step(self, beta: complex):
        r = float(self.r_per_step[self.t])
        sqrt_r = math.sqrt(r)
        scale = self.a_rem / self.alpha if self.alpha > 0 else 1.0

        # true click sample
        a_true = sqrt_r * (self.alphabet[self.m] * scale) - beta
        p1_true = self._click_prob(a_true)
        z = 1 if self.rng.random() < p1_true else 0

        # Bayes update
        L = np.empty(self.K, dtype=np.float64)
        for j in range(self.K):
            a_j = sqrt_r * (self.alphabet[j] * scale) - beta
            p1_j = self._click_prob(a_j)
            L[j] = p1_j if z == 1 else (1.0 - p1_j)

        self.p = self.p * L
        self.p = self.p / (self.p.sum() + 1e-12)

        # update remaining amplitude and time
        self.a_rem = math.sqrt(max(0.0, 1.0 - r)) * self.a_rem
        self.t += 1
        done = self.t >= self.T

        if done:
            self.guess = int(np.argmax(self.p))
            self.guess_correct = (self.guess == self.m)

        return (self.get_obs() if not done else None), z, done

    def final_reward(self) -> float:
        if self.guess is None:
            self.guess = int(np.argmax(self.p))
            self.guess_correct = (self.guess == self.m)
        return 1.0 if self.guess_correct else 0.0


# ============================================================
# 2) State featurization
# ============================================================
def build_obs(p: np.ndarray, t: int, T: int, a_rem: float, alpha: float, r_t: float) -> np.ndarray:
    # log-posterior is usually easier than raw posterior near simplex corners
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


# ============================================================
# 3) Networks: Policy + Value
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, act_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, act_dim),
        )
        self._init_small_output()

    def _init_small_output(self):
        # small final layer => initial actions near zero
        last = self.net[-1]
        nn.init.uniform_(last.weight, -1e-3, 1e-3)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)

    def act(self, x):
        # normalized action in [-1,1]^2
        return torch.tanh(self.forward(x))


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # output success probability in [0,1]
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ============================================================
# 4) AlphaZero-lite local search without handcrafted decision tree
#    policy proposal + weak candidate search + value scoring
# ============================================================
@torch.no_grad()
def one_step_value_score(env: KPSKEnv, value_net: ValueNet, p: np.ndarray,
                         a_rem: float, t: int, beta: complex, device: str) -> float:
    """
    Score beta by expected leaf value after one measurement outcome.
    If t is last step, use exact MAP success after the update.
    """
    K = len(p)
    r = float(env.r_per_step[t])
    sqrt_r = math.sqrt(r)
    scale = a_rem / env.alpha if env.alpha > 0 else 1.0

    p1 = np.zeros(K, dtype=np.float64)
    for m in range(K):
        a_eff = sqrt_r * (env.alphabet[m] * scale) - beta
        p1[m] = env._click_prob(a_eff)

    p_z1 = float(np.dot(p, p1))

    # posterior given click
    num1 = p * p1
    post1 = num1 / (num1.sum() + 1e-12)

    # posterior given no click
    num0 = p * (1.0 - p1)
    post0 = num0 / (num0.sum() + 1e-12)

    if t == env.T - 1:
        v1 = float(np.max(post1))
        v0 = float(np.max(post0))
    else:
        a_rem_next = math.sqrt(max(0.0, 1.0 - r)) * a_rem
        obs1 = build_obs(post1, t + 1, env.T, a_rem_next, env.alpha, env.r_per_step[t + 1])
        obs0 = build_obs(post0, t + 1, env.T, a_rem_next, env.alpha, env.r_per_step[t + 1])
        x1 = torch.tensor(obs1, dtype=torch.float32, device=device).unsqueeze(0)
        x0 = torch.tensor(obs0, dtype=torch.float32, device=device).unsqueeze(0)
        v1 = float(value_net(x1).item())
        v0 = float(value_net(x0).item())

    return p_z1 * v1 + (1.0 - p_z1) * v0


@torch.no_grad()
def local_search_action(env: KPSKEnv, policy_net: PolicyNet, value_net: ValueNet,
                        obs: np.ndarray, device: str,
                        n_random: int = 12,
                        n_phase_candidates: int = 8,
                        sigma_local: float = 0.20,
                        eps_explore: float = 0.10) -> np.ndarray:
    """
    No decision tree required.
    1) policy proposes one normalized action a_prop
    2) generate local candidates around it + structured phase candidates
    3) score each candidate with one-step expected value
    4) pick best candidate (with small epsilon exploration)
    """
    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    a_prop = policy_net.act(x).squeeze(0).cpu().numpy()  # shape (2,)

    # candidate set in normalized action space [-1,1]^2
    cands = [a_prop.copy(), np.zeros(2, dtype=np.float32)]

    # local Gaussian perturbations around proposal
    for _ in range(n_random):
        cand = a_prop + np.random.normal(loc=0.0, scale=sigma_local, size=2)
        cand = np.clip(cand, -1.0, 1.0)
        cands.append(cand.astype(np.float32))

    # structured phase-aligned candidates (no handcrafted tree, only symmetry prior)
    radii = [0.25, 0.50, 0.75, 1.00]
    phase_count = max(n_phase_candidates, env.K)
    for phi in np.linspace(0, 2 * np.pi, phase_count, endpoint=False):
        for rad in radii:
            cands.append(np.array([rad * math.cos(phi), rad * math.sin(phi)], dtype=np.float32))

    # occasional random global exploration
    if np.random.rand() < eps_explore:
        for _ in range(6):
            cands.append(np.random.uniform(low=-1.0, high=1.0, size=2).astype(np.float32))

    # deduplicate roughly
    uniq = []
    seen = set()
    for c in cands:
        key = (round(float(c[0]), 3), round(float(c[1]), 3))
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    cands = uniq

    p = env.p.copy()
    t = int(env.t)
    a_rem = float(env.a_rem)

    best_a = cands[0]
    best_score = -1e18
    for a in cands:
        beta = env.normalized_action_to_beta(a)
        score = one_step_value_score(env, value_net, p, a_rem, t, beta, device)
        if score > best_score:
            best_score = score
            best_a = a

    return best_a.astype(np.float32)


# ============================================================
# 5) Data collection for self-improvement
# ============================================================
@torch.no_grad()
def collect_self_improvement_dataset(policy_net: PolicyNet,
                                     value_net: ValueNet,
                                     train_alphas: List[float],
                                     K: int = 4,
                                     T: int = 6,
                                     eta: float = 1.0,
                                     p_dark: float = 0.0,
                                     episodes: int = 512,
                                     seed: int = 0,
                                     device: str = "cpu"):
    rng = np.random.default_rng(seed)
    X, A, Y = [], [], []

    for ep in range(episodes):
        alpha = float(rng.choice(train_alphas))
        env = KPSKEnv(K=K, alpha=alpha, eta=eta, p_dark=p_dark, T=T,
                      seed=int(rng.integers(0, 10_000_000)))
        obs = env.reset_episode()
        episode_states = []
        episode_actions = []

        done = False
        while not done:
            a_exec = local_search_action(env, policy_net, value_net, obs, device=device)
            beta = env.normalized_action_to_beta(a_exec)

            episode_states.append(obs.copy())
            episode_actions.append(a_exec.copy())

            obs, _, done = env.step(beta)

        y = float(env.final_reward())
        for s, a in zip(episode_states, episode_actions):
            X.append(s)
            A.append(a)
            Y.append(y)

    X = np.asarray(X, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    return X, A, Y


# ============================================================
# 6) Training
# ============================================================
@dataclass
class TrainConfig:
    K: int = 4
    T: int = 6
    eta: float = 1.0
    p_dark: float = 0.0
    train_alphas: Tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2)
    hidden: int = 256
    outer_rounds: int = 25
    collect_episodes: int = 512
    batch_size: int = 256
    train_epochs: int = 8
    policy_lr: float = 1e-3
    value_lr: float = 1e-3
    policy_weight: float = 1.0
    value_weight: float = 1.0
    weight_terminal_only_last_steps: bool = False
    device: str = "cpu"
    seed: int = 0


def train_no_teacher_alpha_zero_lite(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    obs_dim = cfg.K + 4
    device = cfg.device

    policy_net = PolicyNet(obs_dim=obs_dim, hidden=cfg.hidden).to(device)
    value_net = ValueNet(obs_dim=obs_dim, hidden=cfg.hidden).to(device)

    opt_policy = torch.optim.Adam(policy_net.parameters(), lr=cfg.policy_lr)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    hist = {"round": [], "policy_loss": [], "value_loss": [], "dataset_acc": []}

    for rd in range(cfg.outer_rounds):
        # 1) collect improved targets using current policy + current value + local search
        X, A, Y = collect_self_improvement_dataset(
            policy_net=policy_net,
            value_net=value_net,
            train_alphas=list(cfg.train_alphas),
            K=cfg.K,
            T=cfg.T,
            eta=cfg.eta,
            p_dark=cfg.p_dark,
            episodes=cfg.collect_episodes,
            seed=cfg.seed + 1000 * rd,
            device=device,
        )

        dataset_acc = float(np.mean(Y))

        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(A, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
        )
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

        # 2) supervised improvement: policy imitates search-improved action,
        #    value predicts final success probability
        policy_net.train()
        value_net.train()

        mean_pl, mean_vl, n_batches = 0.0, 0.0, 0
        for _ in range(cfg.train_epochs):
            for xb, ab, yb in dl:
                xb = xb.to(device)
                ab = ab.to(device)
                yb = yb.to(device)

                # policy loss in normalized action space
                pred_a = policy_net.act(xb)
                # SmoothL1 is more robust than plain MSE around sharp decision boundaries
                loss_policy = F.smooth_l1_loss(pred_a, ab)

                # value loss
                pred_v = value_net(xb)
                loss_value = F.mse_loss(pred_v, yb)

                opt_policy.zero_grad(set_to_none=True)
                (cfg.policy_weight * loss_policy).backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                opt_policy.step()

                opt_value.zero_grad(set_to_none=True)
                (cfg.value_weight * loss_value).backward()
                nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
                opt_value.step()

                mean_pl += float(loss_policy.item())
                mean_vl += float(loss_value.item())
                n_batches += 1

        hist["round"].append(rd)
        hist["policy_loss"].append(mean_pl / max(1, n_batches))
        hist["value_loss"].append(mean_vl / max(1, n_batches))
        hist["dataset_acc"].append(dataset_acc)

        print(f"[round {rd:02d}] dataset_acc={dataset_acc:.4f}  "
              f"policy_loss={hist['policy_loss'][-1]:.6f}  value_loss={hist['value_loss'][-1]:.6f}")

    return policy_net, value_net, hist


# ============================================================
# 7) Evaluation
# ============================================================
@torch.no_grad()
def eval_policy_only(policy_net: PolicyNet, alpha: float,
                     K: int = 4, T: int = 6,
                     eta: float = 1.0, p_dark: float = 0.0,
                     trials: int = 2000, seed: int = 123, device: str = "cpu") -> float:
    """Fast inference: use policy only, no local search."""
    correct = 0.0
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        env = KPSKEnv(K=K, alpha=alpha, eta=eta, p_dark=p_dark, T=T,
                      seed=int(rng.integers(0, 10_000_000)))
        obs = env.reset_episode()
        done = False
        while not done:
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = policy_net.act(x).squeeze(0).cpu().numpy()
            beta = env.normalized_action_to_beta(a)
            obs, _, done = env.step(beta)
        correct += env.final_reward()
    return correct / trials


@torch.no_grad()
def eval_search_guided(policy_net: PolicyNet, value_net: ValueNet, alpha: float,
                       K: int = 4, T: int = 6,
                       eta: float = 1.0, p_dark: float = 0.0,
                       trials: int = 2000, seed: int = 123, device: str = "cpu") -> float:
    """Inference with local search guided by policy+value."""
    correct = 0.0
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        env = KPSKEnv(K=K, alpha=alpha, eta=eta, p_dark=p_dark, T=T,
                      seed=int(rng.integers(0, 10_000_000)))
        obs = env.reset_episode()
        done = False
        while not done:
            a_exec = local_search_action(env, policy_net, value_net, obs, device=device,
                                         n_random=12, n_phase_candidates=8,
                                         sigma_local=0.15, eps_explore=0.0)
            beta = env.normalized_action_to_beta(a_exec)
            obs, _, done = env.step(beta)
        correct += env.final_reward()
    return correct / trials


# ============================================================
# 8) Main
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig(
        K=4,
        T=6,
        eta=1.0,
        p_dark=0.0,
        train_alphas=(0.4, 0.6, 0.8, 1.0, 1.2),
        hidden=256,
        outer_rounds=20,
        collect_episodes=512,
        batch_size=256,
        train_epochs=8,
        policy_lr=1e-3,
        value_lr=1e-3,
        device=device,
        seed=42,
    )

    policy_net, value_net, hist = train_no_teacher_alpha_zero_lite(cfg)

    print("\nEvaluation:")
    eval_alphas = [0.4, 0.6, 0.8, 1.0, 1.2]
    for a in eval_alphas:
        acc_pi = eval_policy_only(policy_net, alpha=a, K=cfg.K, T=cfg.T,
                                  eta=cfg.eta, p_dark=cfg.p_dark,
                                  trials=2000, device=cfg.device)
        acc_guided = eval_search_guided(policy_net, value_net, alpha=a, K=cfg.K, T=cfg.T,
                                        eta=cfg.eta, p_dark=cfg.p_dark,
                                        trials=2000, device=cfg.device)
        print(f"alpha={a:.2f}  policy_only={acc_pi:.4f}  search_guided={acc_guided:.4f}")


if __name__ == "__main__":
    main()
