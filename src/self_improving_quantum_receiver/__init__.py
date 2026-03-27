"""
Top-level package for self_improving_quantum_receiver.
"""

from .methods import (
    TrainConfig,
    PolicyNet,
    ValueNet,
    train_no_teacher_alpha_zero_lite,
    eval_policy_only,
    eval_search_guided,
)

from .baselines import (
    helstrom_kpsk_pure_closed_form,
    DolinarLikePolicy,
)

__all__ = [
    "TrainConfig",
    "PolicyNet",
    "ValueNet",
    "train_no_teacher_alpha_zero_lite",
    "eval_policy_only",
    "eval_search_guided",
    "helstrom_kpsk_pure_closed_form",
    "DolinarLikePolicy",
]