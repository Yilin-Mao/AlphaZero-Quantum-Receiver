import json
import hashlib
import math
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from self_improving_quantum_receiver.environments import make_env


# ============================================================
# 1) Networks: Policy + Value
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
        last = self.net[-1]
        nn.init.uniform_(last.weight, -1e-3, 1e-3)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        return self.net(x)

    def act(self, x):
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
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ============================================================
# 2) Search / value scoring
# ============================================================
@torch.no_grad()
def one_step_value_score(env, value_net: ValueNet, p: np.ndarray,
                         a_rem: float, t: int, beta: complex, device: str) -> float:
    K = len(p)
    r = float(env.r_per_step[t])
    sqrt_r = math.sqrt(r)
    scale = a_rem / env.alpha if env.alpha > 0 else 1.0

    symbol_fields = env.get_symbol_fields(t, scale=scale)

    p1 = np.zeros(K, dtype=np.float64)
    for m in range(K):
        a_eff = sqrt_r * symbol_fields[m] - beta
        p1[m] = env._click_prob(a_eff)

    p_z1 = float(np.dot(p, p1))

    num1 = p * p1
    post1 = num1 / (num1.sum() + 1e-12)

    num0 = p * (1.0 - p1)
    post0 = num0 / (num0.sum() + 1e-12)

    if t == env.T - 1:
        v1 = float(np.max(post1))
        v0 = float(np.max(post0))
    else:
        a_rem_next = math.sqrt(max(0.0, 1.0 - r)) * a_rem

        old_p = env.p.copy()
        old_t = env.t
        old_a_rem = env.a_rem

        # reuse env's own observation logic for consistency across envs
        env.p = post1.copy()
        env.t = t + 1
        env.a_rem = a_rem_next
        obs1 = env.get_obs()

        env.p = post0.copy()
        env.t = t + 1
        env.a_rem = a_rem_next
        obs0 = env.get_obs()

        env.p = old_p
        env.t = old_t
        env.a_rem = old_a_rem

        x1 = torch.tensor(obs1, dtype=torch.float32, device=device).unsqueeze(0)
        x0 = torch.tensor(obs0, dtype=torch.float32, device=device).unsqueeze(0)
        v1 = float(value_net(x1).item())
        v0 = float(value_net(x0).item())

    return p_z1 * v1 + (1.0 - p_z1) * v0


@torch.no_grad()
def local_search_action(env, policy_net: PolicyNet, value_net: ValueNet,
                        obs: np.ndarray, device: str,
                        n_random: int = 12,
                        n_phase_candidates: int = 8,
                        sigma_local: float = 0.20,
                        eps_explore: float = 0.10) -> np.ndarray:
    x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    a_prop = policy_net.act(x).squeeze(0).cpu().numpy()

    cands = [a_prop.copy(), np.zeros(2, dtype=np.float32)]

    for _ in range(n_random):
        cand = a_prop + np.random.normal(loc=0.0, scale=sigma_local, size=2)
        cand = np.clip(cand, -1.0, 1.0)
        cands.append(cand.astype(np.float32))

    radii = [0.25, 0.50, 0.75, 1.00]
    phase_count = max(n_phase_candidates, env.K)
    for phi in np.linspace(0, 2 * np.pi, phase_count, endpoint=False):
        for rad in radii:
            cands.append(np.array([rad * math.cos(phi), rad * math.sin(phi)], dtype=np.float32))

    if np.random.rand() < eps_explore:
        for _ in range(6):
            cands.append(np.random.uniform(low=-1.0, high=1.0, size=2).astype(np.float32))

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
# 3) Data collection
# ============================================================
@torch.no_grad()
def collect_self_improvement_dataset(policy_net: PolicyNet,
                                     value_net: ValueNet,
                                     train_alphas: List[float],
                                     env_name: str = "standard_kpsk",
                                     env_kwargs: Optional[Dict[str, Any]] = None,
                                     K: int = 4,
                                     T: int = 6,
                                     eta: float = 1.0,
                                     p_dark: float = 0.0,
                                     episodes: int = 512,
                                     seed: int = 0,
                                     device: str = "cpu"):
    rng = np.random.default_rng(seed)
    X, A, Y = [], [], []
    env_kwargs = {} if env_kwargs is None else dict(env_kwargs)

    for _ in range(episodes):
        alpha = float(rng.choice(train_alphas))
        env = make_env(
            env_name=env_name,
            K=K,
            alpha=alpha,
            eta=eta,
            p_dark=p_dark,
            T=T,
            seed=int(rng.integers(0, 10_000_000)),
            **env_kwargs,
        )
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

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(A, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
    )


# ============================================================
# 4) Config + checkpointing
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

    env_name: str = "standard_kpsk"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    checkpoint_root: str = "results/checkpoints"
    experiment_name: str = "alpha_zero_kpsk"
    auto_resume: bool = True
    force_retrain: bool = False
    save_every_round: bool = False


def _format_float_for_name(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace(".", "p").replace("-", "m")


def _alphas_to_name(train_alphas: Tuple[float, ...]) -> str:
    return "-".join(_format_float_for_name(a) for a in train_alphas)


def _env_kwargs_to_name(env_kwargs: Dict[str, Any]) -> str:
    if not env_kwargs:
        return "default"
    payload = json.dumps(env_kwargs, sort_keys=True, separators=(",", ":"))
    short_hash = hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]
    return f"cfg-{short_hash}"


def make_checkpoint_stem(cfg: TrainConfig) -> str:
    env_tag = _env_kwargs_to_name(cfg.env_kwargs)

    train_alpha_tag = hashlib.md5(
        json.dumps(list(cfg.train_alphas)).encode("utf-8")
    ).hexdigest()[:8]

    parts = [
        cfg.experiment_name,
        cfg.env_name,
        env_tag,
        f"K{cfg.K}",
        f"T{cfg.T}",
        f"h{cfg.hidden}",
        f"or{cfg.outer_rounds}",
        f"ce{cfg.collect_episodes}",
        f"bs{cfg.batch_size}",
        f"ep{cfg.train_epochs}",
        f"seed{cfg.seed}",
        f"a{train_alpha_tag}",
    ]
    return "__".join(parts)


def get_checkpoint_dir(cfg: TrainConfig) -> Path:
    path = Path(cfg.checkpoint_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoint_path(cfg: TrainConfig) -> Path:
    return get_checkpoint_dir(cfg) / f"{make_checkpoint_stem(cfg)}.pt"


def save_checkpoint(cfg: TrainConfig, policy_net, value_net, hist: dict, obs_dim: int, path: Optional[Path] = None):
    if path is None:
        path = get_checkpoint_path(cfg)
    payload = {
        "config": asdict(cfg),
        "policy_state_dict": policy_net.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "hist": hist,
        "obs_dim": obs_dim,
    }
    torch.save(payload, path)


def load_checkpoint(cfg: TrainConfig, path: Optional[Path] = None, map_location: Optional[str] = None):
    if path is None:
        path = get_checkpoint_path(cfg)
    if map_location is None:
        map_location = cfg.device

    payload = torch.load(path, map_location=map_location)
    obs_dim = int(payload["obs_dim"])

    policy_net = PolicyNet(obs_dim=obs_dim, hidden=cfg.hidden).to(cfg.device)
    value_net = ValueNet(obs_dim=obs_dim, hidden=cfg.hidden).to(cfg.device)
    policy_net.load_state_dict(payload["policy_state_dict"])
    value_net.load_state_dict(payload["value_state_dict"])

    hist = payload.get("hist", {"round": [], "policy_loss": [], "value_loss": [], "dataset_acc": []})
    policy_net.eval()
    value_net.eval()
    return policy_net, value_net, hist, obs_dim


def maybe_load_existing_checkpoint(cfg: TrainConfig):
    path = get_checkpoint_path(cfg)
    if cfg.auto_resume and (not cfg.force_retrain) and path.exists():
        print(f"[checkpoint] loading existing checkpoint: {path}")
        return load_checkpoint(cfg, path=path)
    return None


# ============================================================
# 5) Training
# ============================================================
def train_no_teacher_alpha_zero_lite(cfg: TrainConfig):
    loaded = maybe_load_existing_checkpoint(cfg)
    if loaded is not None:
        return loaded

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    probe_env = make_env(
        env_name=cfg.env_name,
        K=cfg.K,
        alpha=float(cfg.train_alphas[0]),
        eta=cfg.eta,
        p_dark=cfg.p_dark,
        T=cfg.T,
        seed=cfg.seed,
        **cfg.env_kwargs,
    )
    obs_dim = probe_env.get_obs_dim()
    device = cfg.device

    policy_net = PolicyNet(obs_dim=obs_dim, hidden=cfg.hidden).to(device)
    value_net = ValueNet(obs_dim=obs_dim, hidden=cfg.hidden).to(device)

    opt_policy = torch.optim.Adam(policy_net.parameters(), lr=cfg.policy_lr)
    opt_value = torch.optim.Adam(value_net.parameters(), lr=cfg.value_lr)

    hist = {"round": [], "policy_loss": [], "value_loss": [], "dataset_acc": []}
    final_ckpt_path = get_checkpoint_path(cfg)

    print(f"[checkpoint] training from scratch, will save to: {final_ckpt_path}")

    for rd in range(cfg.outer_rounds):
        X, A, Y = collect_self_improvement_dataset(
            policy_net=policy_net,
            value_net=value_net,
            train_alphas=list(cfg.train_alphas),
            env_name=cfg.env_name,
            env_kwargs=cfg.env_kwargs,
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

        policy_net.train()
        value_net.train()

        mean_pl, mean_vl, n_batches = 0.0, 0.0, 0
        for _ in range(cfg.train_epochs):
            for xb, ab, yb in dl:
                xb = xb.to(device)
                ab = ab.to(device)
                yb = yb.to(device)

                pred_a = policy_net.act(xb)
                loss_policy = F.smooth_l1_loss(pred_a, ab)

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
              f"policy_loss={hist['policy_loss'][-1]:.6f}  "
              f"value_loss={hist['value_loss'][-1]:.6f}")

        if cfg.save_every_round:
            round_ckpt = final_ckpt_path.with_name(final_ckpt_path.stem + f"__round{rd:02d}.pt")
            save_checkpoint(cfg, policy_net, value_net, hist, obs_dim=obs_dim, path=round_ckpt)

    policy_net.eval()
    value_net.eval()
    save_checkpoint(cfg, policy_net, value_net, hist, obs_dim=obs_dim, path=final_ckpt_path)
    print(f"[checkpoint] saved final checkpoint: {final_ckpt_path}")

    return policy_net, value_net, hist, obs_dim


# ============================================================
# 6) Evaluation
# ============================================================
@torch.no_grad()
def eval_policy_only(policy_net: PolicyNet, alpha: float,
                     env_name: str = "standard_kpsk",
                     env_kwargs: Optional[Dict[str, Any]] = None,
                     K: int = 4, T: int = 6,
                     eta: float = 1.0, p_dark: float = 0.0,
                     trials: int = 2000, seed: int = 123, device: str = "cpu") -> float:
    correct = 0.0
    rng = np.random.default_rng(seed)
    env_kwargs = {} if env_kwargs is None else dict(env_kwargs)

    for _ in range(trials):
        env = make_env(
            env_name=env_name,
            K=K,
            alpha=alpha,
            eta=eta,
            p_dark=p_dark,
            T=T,
            seed=int(rng.integers(0, 10_000_000)),
            **env_kwargs,
        )
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
                       env_name: str = "standard_kpsk",
                       env_kwargs: Optional[Dict[str, Any]] = None,
                       K: int = 4, T: int = 6,
                       eta: float = 1.0, p_dark: float = 0.0,
                       trials: int = 2000, seed: int = 123, device: str = "cpu") -> float:
    correct = 0.0
    rng = np.random.default_rng(seed)
    env_kwargs = {} if env_kwargs is None else dict(env_kwargs)

    for _ in range(trials):
        env = make_env(
            env_name=env_name,
            K=K,
            alpha=alpha,
            eta=eta,
            p_dark=p_dark,
            T=T,
            seed=int(rng.integers(0, 10_000_000)),
            **env_kwargs,
        )
        obs = env.reset_episode()
        done = False
        while not done:
            a_exec = local_search_action(
                env, policy_net, value_net, obs, device=device,
                n_random=12, n_phase_candidates=8,
                sigma_local=0.15, eps_explore=0.0
            )
            beta = env.normalized_action_to_beta(a_exec)
            obs, _, done = env.step(beta)
        correct += env.final_reward()

    return correct / trials


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = TrainConfig(
        K=4,
        T=12,
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
        env_name="chirped_kpsk",
        env_kwargs={
            "chirp_amp_1": 0.42,
            "chirp_freq_1": 3.0,
            "chirp_amp_2": 0.18,
            "chirp_freq_2": 7.0,
            "chirp_phase_2": 0.3,
            "energy_omega": 3.5,
            "energy_floor": 0.18,
            "use_oscillatory_schedule": True,
        },
    )

    policy_net, value_net, hist, obs_dim = train_no_teacher_alpha_zero_lite(cfg)

    print("\nEvaluation:")
    for a in [0.4, 0.6, 0.8, 1.0, 1.2]:
        acc_pi = eval_policy_only(
            policy_net, alpha=a,
            env_name=cfg.env_name, env_kwargs=cfg.env_kwargs,
            K=cfg.K, T=cfg.T, eta=cfg.eta, p_dark=cfg.p_dark,
            trials=2000, device=cfg.device,
        )
        acc_guided = eval_search_guided(
            policy_net, value_net, alpha=a,
            env_name=cfg.env_name, env_kwargs=cfg.env_kwargs,
            K=cfg.K, T=cfg.T, eta=cfg.eta, p_dark=cfg.p_dark,
            trials=2000, device=cfg.device,
        )
        print(f"alpha={a:.2f}  policy_only={acc_pi:.4f}  search_guided={acc_guided:.4f}")


if __name__ == "__main__":
    main()