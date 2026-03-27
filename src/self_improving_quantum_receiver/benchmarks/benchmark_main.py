import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import hashlib
import json

from self_improving_quantum_receiver.methods import (
    TrainConfig,
    PolicyNet,
    ValueNet,
    train_no_teacher_alpha_zero_lite,
    eval_policy_only,
    eval_search_guided,
)
from self_improving_quantum_receiver.environments import make_env
from self_improving_quantum_receiver.baselines import (
    helstrom_kpsk_pure_closed_form,
    DolinarLikePolicy,
)


# ============================================================
# 0) Paths / tags / utils
# ============================================================
def _format_tag_value(v):
    if isinstance(v, float):
        s = f"{v:.6g}"
        return s.replace(".", "p").replace("-", "m")
    return str(v).replace("/", "-").replace(" ", "_")


def make_experiment_tag(cfg, extra_tag: str = ""):
    env_payload = json.dumps(getattr(cfg, "env_kwargs", {}), sort_keys=True, separators=(",", ":"))
    env_hash = hashlib.md5(env_payload.encode("utf-8")).hexdigest()[:10]

    parts = [
        f"env-{cfg.env_name}",
        f"cfg-{env_hash}",
        f"K{cfg.K}",
        f"T{cfg.T}",
        f"h{cfg.hidden}",
        f"or{cfg.outer_rounds}",
        f"seed{cfg.seed}",
    ]

    if extra_tag:
        parts.append(extra_tag)

    return "__".join(parts)


def ensure_dirs(experiment_tag: str):
    root = Path("results") / experiment_tag
    fig_dir = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return root, fig_dir


def count_params(module) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def error_rate(acc: float) -> float:
    return 1.0 - acc


def safe_log10_error(acc: float, eps: float = 1e-12) -> float:
    return math.log10(max(1.0 - acc, eps))


def save_csv(rows, save_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_lines(x, series_dict, ylabel, title, save_path, include_keys=None):
    plt.figure(figsize=(8, 5))
    keys = include_keys if include_keys is not None else list(series_dict.keys())
    for k in keys:
        plt.plot(x, series_dict[k], marker="o", label=k)
    plt.xlabel("alpha")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


# ============================================================
# 1) Complexity helpers
# ============================================================
def mlp_param_count(in_dim: int, hidden: int, out_dim: int) -> int:
    return (
        (in_dim * hidden + hidden) +
        (hidden * hidden + hidden) +
        (hidden * hidden + hidden) +
        (hidden * out_dim + out_dim)
    )


def estimate_policy_value_params(obs_dim: int, hidden: int):
    policy_params = mlp_param_count(obs_dim, hidden, 2)
    value_params = mlp_param_count(obs_dim, hidden, 1)
    return policy_params, value_params, policy_params + value_params


def geometric_tree_nodes(branching_factor: int, depth: int) -> int:
    b = int(branching_factor)
    d = int(depth)
    if b <= 1:
        return d + 1
    return (b ** (d + 1) - 1) // (b - 1)


def dolinar_structural_stats(policy, T: int):
    n_rho_coarse = len(policy.beta_grid_coarse)
    n_phi_coarse = len(policy.phase_grid_coarse)
    n_rho_refine = len(policy.beta_grid_refine)
    n_phi_refine = len(policy.phase_grid_refine)

    coarse_candidates = n_rho_coarse * n_phi_coarse
    refine_candidates = n_rho_refine * n_phi_refine
    candidates_per_step = coarse_candidates + refine_candidates

    structural_param_count = (
        n_rho_coarse + n_phi_coarse + n_rho_refine + n_phi_refine
    )

    return {
        "coarse_candidates": coarse_candidates,
        "refine_candidates": refine_candidates,
        "search_cost_per_step": candidates_per_step,
        "total_search_cost_per_episode": T * candidates_per_step,
        "structural_param_count": structural_param_count,
    }


def guided_search_structural_stats(K: int,
                                   T: int,
                                   n_random: int = 12,
                                   n_phase_candidates: int = 8,
                                   use_zero_action: bool = True,
                                   use_policy_proposal: bool = True,
                                   use_global_explore: bool = False,
                                   global_explore_num: int = 6):
    base = 0
    if use_policy_proposal:
        base += 1
    if use_zero_action:
        base += 1
    base += n_random
    base += max(n_phase_candidates, K) * 4
    if use_global_explore:
        base += global_explore_num

    return {
        "search_cost_per_step_nominal": base,
        "total_search_cost_per_episode_nominal": T * base,
        "structural_param_count": 0,
    }


def get_complexity_profile_for_learned(obs_dim: int, K: int, T: int, hidden: int,
                                       n_random: int = 12,
                                       n_phase_candidates: int = 8):
    policy_params, value_params, total_params = estimate_policy_value_params(obs_dim, hidden)
    search_stats = guided_search_structural_stats(
        K=K, T=T,
        n_random=n_random,
        n_phase_candidates=n_phase_candidates,
        use_zero_action=True,
        use_policy_proposal=True,
        use_global_explore=False,
    )
    return {
        "learned_param_count": total_params,
        "policy_params": policy_params,
        "value_params": value_params,
        "structural_param_count": search_stats["structural_param_count"],
        "search_cost_per_step_nominal": search_stats["search_cost_per_step_nominal"],
        "total_search_cost_per_episode_nominal": search_stats["total_search_cost_per_episode_nominal"],
        "effective_tree_nodes": 0,
        "param_scaling_wrt_T": "O(1)",
        "inference_scaling_wrt_T": "O(T)",
    }


def get_complexity_profile_for_dolinar(env, T: int):
    pol = DolinarLikePolicy(env)
    stats = dolinar_structural_stats(pol, T=T)
    return {
        "learned_param_count": 0,
        "policy_params": 0,
        "value_params": 0,
        "structural_param_count": stats["structural_param_count"],
        "search_cost_per_step_nominal": stats["search_cost_per_step"],
        "total_search_cost_per_episode_nominal": stats["total_search_cost_per_episode"],
        "effective_tree_nodes": 0,
        "param_scaling_wrt_T": "O(1)",
        "inference_scaling_wrt_T": "O(T)",
    }


def get_complexity_profile_for_explicit_tree(T: int, branching_factor: int):
    return {
        "learned_param_count": 0,
        "policy_params": 0,
        "value_params": 0,
        "structural_param_count": geometric_tree_nodes(branching_factor, T),
        "search_cost_per_step_nominal": branching_factor,
        "total_search_cost_per_episode_nominal": None,
        "effective_tree_nodes": geometric_tree_nodes(branching_factor, T),
        "param_scaling_wrt_T": "O(b^T)",
        "inference_scaling_wrt_T": "O(T) or O(tree traversal)",
    }


# ============================================================
# 2) Non-learned baselines
# ============================================================
# def act_zero_beta(env) -> complex:
#     return 0.0 + 0.0j


# def act_random(env, rng: np.random.Generator) -> complex:
#     a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
#     return env.normalized_action_to_beta(a)


# def act_greedy_map_cancel(env) -> complex:
#     j = int(np.argmax(env.p))
#     r = float(env.r_per_step[env.t])
#     sqrt_r = math.sqrt(r)
#     scale = env.a_rem / env.alpha if env.alpha > 0 else 1.0
#     symbol_fields = env.get_symbol_fields(env.t, scale=scale)
#     return sqrt_r * symbol_fields[j]


# ============================================================
# 3) Generic rollout evaluators
# ============================================================
def eval_generic_baseline(alpha: float,
                          method_name: str,
                          action_fn,
                          env_name: str = "chirped_kpsk",
                          env_kwargs=None,
                          K: int = 4,
                          T: int = 6,
                          eta: float = 1.0,
                          p_dark: float = 0.0,
                          trials: int = 1000,
                          seed: int = 123):
    rng = np.random.default_rng(seed)
    correct = 0.0
    total_trial_time = 0.0

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
        env.reset_episode()
        done = False

        t0 = time.perf_counter()
        while not done:
            beta = action_fn(env, rng) if method_name == "random_action" else action_fn(env)
            _, _, done = env.step(beta)
        total_trial_time += time.perf_counter() - t0
        correct += env.final_reward()

    acc = correct / trials
    avg_trial_t = total_trial_time / trials
    avg_step_t = avg_trial_t / T

    return {
        "accuracy": acc,
        "avg_time_per_trial_sec": avg_trial_t,
        "avg_time_per_step_sec": avg_step_t,
        "search_cost_per_step": 1.0,
    }


def eval_dolinar_like(alpha: float,
                      env_name: str = "chirped_kpsk",
                      env_kwargs=None,
                      K: int = 4,
                      T: int = 6,
                      eta: float = 1.0,
                      p_dark: float = 0.0,
                      trials: int = 1000,
                      seed: int = 123):
    rng = np.random.default_rng(seed)
    correct = 0.0
    total_trial_time = 0.0
    total_steps = 0

    env_kwargs = {} if env_kwargs is None else dict(env_kwargs)

    nominal_cost = None

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
        env.reset_episode()
        pol = DolinarLikePolicy(env,
                                beta_grid_coarse=np.linspace(0.0, 1.0, 5),
                                phase_grid_coarse=np.linspace(0.0, 2.0 * math.pi, K, endpoint=False),
                                beta_grid_refine=np.linspace(-1.0, 1.0, 4),
                                phase_grid_refine=np.linspace(-1.0, 1.0, 4))

        if nominal_cost is None:
            stats = dolinar_structural_stats(pol, T=T)
            nominal_cost = float(stats["search_cost_per_step"])

        done = False
        t0 = time.perf_counter()
        while not done:
            beta = pol.act()
            _, _, done = env.step(beta)
            total_steps += 1
        total_trial_time += time.perf_counter() - t0
        correct += env.final_reward()

    acc = correct / trials
    avg_trial_t = total_trial_time / trials
    avg_step_t = avg_trial_t / T

    return {
        "accuracy": acc,
        "avg_time_per_trial_sec": avg_trial_t,
        "avg_time_per_step_sec": avg_step_t,
        "search_cost_per_step": nominal_cost if nominal_cost is not None else 0.0,
    }


def eval_policy_only_custom(policy_net: PolicyNet,
                            alpha: float,
                            env_name: str = "chirped_kpsk",
                            env_kwargs=None,
                            K: int = 4,
                            T: int = 6,
                            eta: float = 1.0,
                            p_dark: float = 0.0,
                            trials: int = 1000,
                            seed: int = 123,
                            device: str = "cpu"):
    t0 = time.perf_counter()
    acc = eval_policy_only(
        policy_net=policy_net,
        alpha=alpha,
        env_name=env_name,
        env_kwargs=env_kwargs,
        K=K,
        T=T,
        eta=eta,
        p_dark=p_dark,
        trials=trials,
        seed=seed,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    avg_trial_t = elapsed / trials
    avg_step_t = avg_trial_t / T

    return {
        "accuracy": acc,
        "avg_time_per_trial_sec": avg_trial_t,
        "avg_time_per_step_sec": avg_step_t,
        "search_cost_per_step": 1.0,
    }


def eval_search_guided_custom(policy_net: PolicyNet,
                              value_net: ValueNet,
                              alpha: float,
                              env_name: str = "chirped_kpsk",
                              env_kwargs=None,
                              K: int = 4,
                              T: int = 6,
                              eta: float = 1.0,
                              p_dark: float = 0.0,
                              trials: int = 1000,
                              seed: int = 123,
                              device: str = "cpu",
                              n_random: int = 12,
                              n_phase_candidates: int = 8):
    t0 = time.perf_counter()
    acc = eval_search_guided(
        policy_net=policy_net,
        value_net=value_net,
        alpha=alpha,
        env_name=env_name,
        env_kwargs=env_kwargs,
        K=K,
        T=T,
        eta=eta,
        p_dark=p_dark,
        trials=trials,
        seed=seed,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    avg_trial_t = elapsed / trials
    avg_step_t = avg_trial_t / T

    nominal_cost = 1 + 1 + n_random + max(n_phase_candidates, K) * 4

    return {
        "accuracy": acc,
        "avg_time_per_trial_sec": avg_trial_t,
        "avg_time_per_step_sec": avg_step_t,
        "search_cost_per_step": float(nominal_cost),
    }


# ============================================================
# 4) Training wrappers
# ============================================================
def train_model_with_timing(cfg: TrainConfig, tag: str):
    t0 = time.perf_counter()
    policy_net, value_net, hist, obs_dim = train_no_teacher_alpha_zero_lite(cfg)
    elapsed = time.perf_counter() - t0

    info = {
        "tag": tag,
        "train_time_sec": elapsed,
        "policy_params": count_params(policy_net),
        "value_params": count_params(value_net),
        "total_params": count_params(policy_net) + count_params(value_net),
        "obs_dim": obs_dim,
        "cfg_hidden": cfg.hidden,
        "cfg_outer_rounds": cfg.outer_rounds,
        "cfg_collect_episodes": cfg.collect_episodes,
        "cfg_batch_size": cfg.batch_size,
        "cfg_train_epochs": cfg.train_epochs,
        "cfg_policy_lr": cfg.policy_lr,
        "cfg_value_lr": cfg.value_lr,
        "env_name": cfg.env_name,
        "env_kwargs": str(cfg.env_kwargs),
        "hist": hist,
    }
    return policy_net, value_net, info


# ============================================================
# 5) Main benchmark
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # choose experiment env here
    # --------------------------------------------------------
    base_env_name = "chirped_kpsk"
    base_env_kwargs = {
        "chirp_amp_1": 0.42,
        "chirp_freq_1": 3.0,
        "chirp_amp_2": 0.18,
        "chirp_freq_2": 7.0,
        "chirp_phase_2": 0.3,
        "energy_omega": 3.5,
        "energy_floor": 0.18,
        "use_oscillatory_schedule": True,
    }

    cfg_small = TrainConfig(
        K=4,
        T=12,
        eta=1.0,
        p_dark=0.0,
        train_alphas=(0.4, 0.6, 0.8, 1.0, 1.2),
        hidden=128,
        outer_rounds=12,
        collect_episodes=256,
        batch_size=128,
        train_epochs=6,
        policy_lr=1e-3,
        value_lr=1e-3,
        device=device,
        seed=42,
        env_name=base_env_name,
        env_kwargs=base_env_kwargs,
        experiment_name="alpha_zero_kpsk_small",
    )

    cfg_full = TrainConfig(
        K=4,
        T=6, # 12
        eta=1.0,
        p_dark=0.0,
        train_alphas=(0.4, 0.6, 0.8, 1.0, 1.2),
        hidden=32, # 256
        outer_rounds=10, # 20
        collect_episodes=512,
        batch_size=256,
        train_epochs=8,
        policy_lr=1e-3,
        value_lr=1e-3,
        device=device,
        seed=42,
        env_name=base_env_name,
        env_kwargs=base_env_kwargs,
        experiment_name="alpha_zero_kpsk_full",
    )

    experiment_tag = make_experiment_tag(cfg_full)
    results_dir, fig_dir = ensure_dirs(experiment_tag)

    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # cfg_small.checkpoint_root = str(checkpoint_dir)
    cfg_full.checkpoint_root = str(checkpoint_dir)

    print("=" * 72)
    print("Training learned baselines ...")
    print("=" * 72)

    # policy_small, value_small, info_small = train_model_with_timing(cfg_small, "small")
    policy_full, value_full, info_full = train_model_with_timing(cfg_full, "full")

    eval_alphas = [0.4, 0.6, 0.8, 1.0, 1.2]
    trials = 1500

    # probe env for complexity accounting
    probe_env = make_env(
        env_name=cfg_full.env_name,
        K=cfg_full.K,
        alpha=eval_alphas[0],
        eta=cfg_full.eta,
        p_dark=cfg_full.p_dark,
        T=cfg_full.T,
        seed=cfg_full.seed,
        **cfg_full.env_kwargs,
    )

    metric_series = {
        "zero_beta": {"acc": [], "logerr": [], "time": [], "cost": []},
        "random_action": {"acc": [], "logerr": [], "time": [], "cost": []},
        "greedy_map_cancel": {"acc": [], "logerr": [], "time": [], "cost": []},
        "dolinar_like": {"acc": [], "logerr": [], "time": [], "cost": []},
        "policy_only_small": {"acc": [], "logerr": [], "time": [], "cost": []},
        "search_guided_small": {"acc": [], "logerr": [], "time": [], "cost": []},
        "policy_only_full": {"acc": [], "logerr": [], "time": [], "cost": []},
        "search_guided_full": {"acc": [], "logerr": [], "time": [], "cost": []},
        "helstrom": {"acc": [], "logerr": [], "time": [], "cost": []},
    }

    rows = []

    def append_row(method_name, family, model_size, alpha, res, complexity):
        acc = res["accuracy"]
        row = {
            "method": method_name,
            "method_family": family,
            "model_size": model_size,
            "env_name": cfg_full.env_name,
            "env_kwargs": str(cfg_full.env_kwargs),
            "alpha": alpha,
            "accuracy": acc,
            "error_rate": error_rate(acc),
            "log10_error_rate": safe_log10_error(acc),
            "avg_time_per_trial_sec": res["avg_time_per_trial_sec"],
            "avg_time_per_step_sec": res["avg_time_per_step_sec"],
            "search_cost_per_step": res["search_cost_per_step"],
            "learned_param_count": complexity["learned_param_count"],
            "policy_params": complexity["policy_params"],
            "value_params": complexity["value_params"],
            "structural_param_count": complexity["structural_param_count"],
            "search_cost_per_step_nominal": complexity["search_cost_per_step_nominal"],
            "total_search_cost_per_episode_nominal": complexity["total_search_cost_per_episode_nominal"],
            "effective_tree_nodes": complexity["effective_tree_nodes"],
            "param_scaling_wrt_T": complexity["param_scaling_wrt_T"],
            "inference_scaling_wrt_T": complexity["inference_scaling_wrt_T"],
        }
        rows.append(row)

        metric_series[method_name]["acc"].append(acc)
        metric_series[method_name]["logerr"].append(safe_log10_error(acc))
        metric_series[method_name]["time"].append(res["avg_time_per_trial_sec"])
        metric_series[method_name]["cost"].append(res["search_cost_per_step"])

    print("\n" + "=" * 72)
    print("Running benchmarks ...")
    print("=" * 72)

    # learned_complexity_small = get_complexity_profile_for_learned(
    #     obs_dim=info_small["obs_dim"],
    #     K=cfg_small.K,
    #     T=cfg_small.T,
    #     hidden=cfg_small.hidden,
    #     n_random=8,
    #     n_phase_candidates=6,
    # )

    learned_complexity_full = get_complexity_profile_for_learned(
        obs_dim=info_full["obs_dim"],
        K=cfg_full.K,
        T=cfg_full.T,
        hidden=cfg_full.hidden,
        n_random=12,
        n_phase_candidates=8,
    )

    dolinar_complexity = get_complexity_profile_for_dolinar(probe_env, T=cfg_full.T)

    for alpha in eval_alphas:
        print(f"\nalpha = {alpha:.2f}")

        # res_zero = eval_generic_baseline(
        #     alpha=alpha,
        #     method_name="zero_beta",
        #     action_fn=act_zero_beta,
        #     env_name=cfg_full.env_name,
        #     env_kwargs=cfg_full.env_kwargs,
        #     K=cfg_full.K, T=cfg_full.T, eta=cfg_full.eta, p_dark=cfg_full.p_dark,
        #     trials=trials, seed=123,
        # )
        # append_row(
        #     "zero_beta", "heuristic", "analytic", alpha, res_zero,
        #     {
        #         "learned_param_count": 0,
        #         "policy_params": 0,
        #         "value_params": 0,
        #         "structural_param_count": 0,
        #         "search_cost_per_step_nominal": 1,
        #         "total_search_cost_per_episode_nominal": cfg_full.T,
        #         "effective_tree_nodes": 0,
        #         "param_scaling_wrt_T": "O(1)",
        #         "inference_scaling_wrt_T": "O(T)",
        #     }
        # )

        # res_rand = eval_generic_baseline(
        #     alpha=alpha,
        #     method_name="random_action",
        #     action_fn=act_random,
        #     env_name=cfg_full.env_name,
        #     env_kwargs=cfg_full.env_kwargs,
        #     K=cfg_full.K, T=cfg_full.T, eta=cfg_full.eta, p_dark=cfg_full.p_dark,
        #     trials=trials, seed=123,
        # )
        # append_row(
        #     "random_action", "heuristic", "analytic", alpha, res_rand,
        #     {
        #         "learned_param_count": 0,
        #         "policy_params": 0,
        #         "value_params": 0,
        #         "structural_param_count": 0,
        #         "search_cost_per_step_nominal": 1,
        #         "total_search_cost_per_episode_nominal": cfg_full.T,
        #         "effective_tree_nodes": 0,
        #         "param_scaling_wrt_T": "O(1)",
        #         "inference_scaling_wrt_T": "O(T)",
        #     }
        # )

        # res_greedy = eval_generic_baseline(
        #     alpha=alpha,
        #     method_name="greedy_map_cancel",
        #     action_fn=act_greedy_map_cancel,
        #     env_name=cfg_full.env_name,
        #     env_kwargs=cfg_full.env_kwargs,
        #     K=cfg_full.K, T=cfg_full.T, eta=cfg_full.eta, p_dark=cfg_full.p_dark,
        #     trials=trials, seed=123,
        # )
        # append_row(
        #     "greedy_map_cancel", "heuristic", "analytic", alpha, res_greedy,
        #     {
        #         "learned_param_count": 0,
        #         "policy_params": 0,
        #         "value_params": 0,
        #         "structural_param_count": 0,
        #         "search_cost_per_step_nominal": 1,
        #         "total_search_cost_per_episode_nominal": cfg_full.T,
        #         "effective_tree_nodes": 0,
        #         "param_scaling_wrt_T": "O(1)",
        #         "inference_scaling_wrt_T": "O(T)",
        #     }
        # )

        res_dol = eval_dolinar_like(
            alpha=alpha,
            env_name=cfg_full.env_name,
            env_kwargs=cfg_full.env_kwargs,
            K=cfg_full.K, T=cfg_full.T, eta=cfg_full.eta, p_dark=cfg_full.p_dark,
            trials=trials, seed=123,
        )
        append_row("dolinar_like", "heuristic", "analytic", alpha, res_dol, dolinar_complexity)

        # res_pi_small = eval_policy_only_custom(
        #     policy_net=policy_small,
        #     alpha=alpha,
        #     env_name=cfg_small.env_name,
        #     env_kwargs=cfg_small.env_kwargs,
        #     K=cfg_small.K, T=cfg_small.T, eta=cfg_small.eta, p_dark=cfg_small.p_dark,
        #     trials=trials, seed=123, device=device,
        # )
        # append_row("policy_only_small", "learned", "small", alpha, res_pi_small, learned_complexity_small)

        # res_sg_small = eval_search_guided_custom(
        #     policy_net=policy_small,
        #     value_net=value_small,
        #     alpha=alpha,
        #     env_name=cfg_small.env_name,
        #     env_kwargs=cfg_small.env_kwargs,
        #     K=cfg_small.K, T=cfg_small.T, eta=cfg_small.eta, p_dark=cfg_small.p_dark,
        #     trials=trials, seed=123, device=device,
        #     n_random=8, n_phase_candidates=6,
        # )
        # append_row("search_guided_small", "learned", "small", alpha, res_sg_small, learned_complexity_small)

        res_pi_full = eval_policy_only_custom(
            policy_net=policy_full,
            alpha=alpha,
            env_name=cfg_full.env_name,
            env_kwargs=cfg_full.env_kwargs,
            K=cfg_full.K, T=cfg_full.T, eta=cfg_full.eta, p_dark=cfg_full.p_dark,
            trials=trials, seed=123, device=device,
        )
        append_row("policy_only_full", "learned", "full", alpha, res_pi_full, learned_complexity_full)

        res_sg_full = eval_search_guided_custom(
            policy_net=policy_full,
            value_net=value_full,
            alpha=alpha,
            env_name=cfg_full.env_name,
            env_kwargs=cfg_full.env_kwargs,
            K=cfg_full.K, T=cfg_full.T, eta=cfg_full.eta, p_dark=cfg_full.p_dark,
            trials=trials, seed=123, device=device,
            n_random=12, n_phase_candidates=8,
        )
        append_row("search_guided_full", "learned", "full", alpha, res_sg_full, learned_complexity_full)

        acc_h = helstrom_kpsk_pure_closed_form(alpha=alpha, K=cfg_full.K)
        res_h = {
            "accuracy": acc_h,
            "avg_time_per_trial_sec": 0.0,
            "avg_time_per_step_sec": 0.0,
            "search_cost_per_step": 0.0,
        }
        append_row(
            "helstrom", "analytic_bound", "analytic", alpha, res_h,
            {
                "learned_param_count": 0,
                "policy_params": 0,
                "value_params": 0,
                "structural_param_count": 0,
                "search_cost_per_step_nominal": 0,
                "total_search_cost_per_episode_nominal": 0,
                "effective_tree_nodes": 0,
                "param_scaling_wrt_T": "N/A",
                "inference_scaling_wrt_T": "N/A",
            }
        )

        # print(f"  zero_beta            acc={res_zero['accuracy']:.4f}")
        # print(f"  random_action        acc={res_rand['accuracy']:.4f}")
        # print(f"  greedy_map_cancel    acc={res_greedy['accuracy']:.4f}")
        print(f"  dolinar_like         acc={res_dol['accuracy']:.4f}")
        # print(f"  policy_only_small    acc={res_pi_small['accuracy']:.4f}")
        # print(f"  search_guided_small  acc={res_sg_small['accuracy']:.4f}")
        print(f"  policy_only_full     acc={res_pi_full['accuracy']:.4f}")
        print(f"  search_guided_full   acc={res_sg_full['accuracy']:.4f}")
        print(f"  helstrom             acc={acc_h:.4f}")

    # --------------------------------------------------------
    # Save csvs
    # --------------------------------------------------------
    save_csv(rows, results_dir / "benchmark_results.csv")

    train_summary_rows = [
        # {
        #     "tag": info_small["tag"],
        #     "env_name": info_small["env_name"],
        #     "env_kwargs": info_small["env_kwargs"],
        #     "obs_dim": info_small["obs_dim"],
        #     "train_time_sec": info_small["train_time_sec"],
        #     "policy_params": info_small["policy_params"],
        #     "value_params": info_small["value_params"],
        #     "total_params": info_small["total_params"],
        #     "cfg_hidden": info_small["cfg_hidden"],
        #     "cfg_outer_rounds": info_small["cfg_outer_rounds"],
        #     "cfg_collect_episodes": info_small["cfg_collect_episodes"],
        #     "cfg_batch_size": info_small["cfg_batch_size"],
        #     "cfg_train_epochs": info_small["cfg_train_epochs"],
        #     "cfg_policy_lr": info_small["cfg_policy_lr"],
        #     "cfg_value_lr": info_small["cfg_value_lr"],
        # },
        {
            "tag": info_full["tag"],
            "env_name": info_full["env_name"],
            "env_kwargs": info_full["env_kwargs"],
            "obs_dim": info_full["obs_dim"],
            "train_time_sec": info_full["train_time_sec"],
            "policy_params": info_full["policy_params"],
            "value_params": info_full["value_params"],
            "total_params": info_full["total_params"],
            "cfg_hidden": info_full["cfg_hidden"],
            "cfg_outer_rounds": info_full["cfg_outer_rounds"],
            "cfg_collect_episodes": info_full["cfg_collect_episodes"],
            "cfg_batch_size": info_full["cfg_batch_size"],
            "cfg_train_epochs": info_full["cfg_train_epochs"],
            "cfg_policy_lr": info_full["cfg_policy_lr"],
            "cfg_value_lr": info_full["cfg_value_lr"],
        },
    ]
    save_csv(train_summary_rows, results_dir / "training_budget_summary.csv")

    hist_rows = []
    # for rd, pl, vl, da in zip(
    #     info_small["hist"]["round"],
    #     info_small["hist"]["policy_loss"],
    #     info_small["hist"]["value_loss"],
    #     info_small["hist"]["dataset_acc"],
    # ):
    #     hist_rows.append({
    #         "tag": "small",
    #         "round": rd,
    #         "policy_loss": pl,
    #         "value_loss": vl,
    #         "dataset_acc": da,
    #     })
    for rd, pl, vl, da in zip(
        info_full["hist"]["round"],
        info_full["hist"]["policy_loss"],
        info_full["hist"]["value_loss"],
        info_full["hist"]["dataset_acc"],
    ):
        hist_rows.append({
            "tag": "full",
            "round": rd,
            "policy_loss": pl,
            "value_loss": vl,
            "dataset_acc": da,
        })
    save_csv(hist_rows, results_dir / "training_history.csv")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plot_lines(
        eval_alphas,
        {k: metric_series[k]["acc"] for k in metric_series},
        ylabel="accuracy",
        title="Accuracy vs alpha",
        save_path=fig_dir / "accuracy_vs_alpha.png",
        include_keys=[
            "dolinar_like", "policy_only_full", "search_guided_full", "helstrom"
        ],
    )

    plot_lines(
        eval_alphas,
        {k: metric_series[k]["logerr"] for k in metric_series},
        ylabel="log10(error rate)",
        title="Log-error vs alpha",
        save_path=fig_dir / "log_error_vs_alpha.png",
        include_keys=[
            "dolinar_like", "policy_only_full", "search_guided_full", "helstrom"

        ],
    )

    plot_lines(
        eval_alphas,
        {k: metric_series[k]["time"] for k in metric_series},
        ylabel="avg time per trial (sec)",
        title="Inference runtime vs alpha",
        save_path=fig_dir / "runtime_vs_alpha.png",
        include_keys=[
            "dolinar_like", "policy_only_full", "search_guided_full", "helstrom"
        ],
    )

    plot_lines(
        eval_alphas,
        {k: metric_series[k]["cost"] for k in metric_series},
        ylabel="search cost per step",
        title="Search budget vs alpha",
        save_path=fig_dir / "search_cost_vs_alpha.png",
        include_keys=[
            "dolinar_like", "policy_only_full", "search_guided_full", "helstrom"
        ],
    )

    plt.figure(figsize=(8, 5))
    # plt.plot(info_small["hist"]["round"], info_small["hist"]["dataset_acc"], marker="o", label="dataset_acc_small")
    plt.plot(info_full["hist"]["round"], info_full["hist"]["dataset_acc"], marker="o", label="dataset_acc_full")
    plt.xlabel("outer round")
    plt.ylabel("dataset_acc")
    plt.title("Training self-improvement curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "training_dataset_acc.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    # plt.plot(info_small["hist"]["round"], info_small["hist"]["policy_loss"], marker="o", label="policy_loss_small")
    plt.plot(info_full["hist"]["round"], info_full["hist"]["policy_loss"], marker="o", label="policy_loss_full")
    # plt.plot(info_small["hist"]["round"], info_small["hist"]["value_loss"], marker="o", label="value_loss_small")
    plt.plot(info_full["hist"]["round"], info_full["hist"]["value_loss"], marker="o", label="value_loss_full")
    plt.xlabel("outer round")
    plt.ylabel("loss")
    plt.title("Training losses")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "training_losses.png", dpi=220)
    plt.close()

    print("\n" + "=" * 72)
    print("Done.")
    print(f"Saved benchmark csv: {results_dir / 'benchmark_results.csv'}")
    print(f"Saved training summary: {results_dir / 'training_budget_summary.csv'}")
    print(f"Saved training history: {results_dir / 'training_history.csv'}")
    print(f"Figures dir: {fig_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()