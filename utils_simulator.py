import numpy as np
from tqdm import tqdm
from typing import Any

from obp.utils import check_bandit_feedback_inputs, convert_to_action_dist
from obp.types import BanditFeedback
from obp.policy.policy_type import PolicyType


def run_bandit_simulation(bandit_feedback: BanditFeedback, policy: Any) -> np.ndarray:
    """Standalone version of OBP dev simulator (works with PyPI OBP)."""

    for key_ in ["action", "position", "reward", "pscore", "context"]:
        if key_ not in bandit_feedback:
            raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")

    check_bandit_feedback_inputs(
        context=bandit_feedback["context"],
        action=bandit_feedback["action"],
        reward=bandit_feedback["reward"],
        position=bandit_feedback["position"],
        pscore=bandit_feedback["pscore"],
    )

    selected_actions_list = []
    realized_rewards = []
    dim_context = bandit_feedback["context"].shape[1]

    for action_, reward_, position_, context_ in tqdm(
        zip(
            bandit_feedback["action"],
            bandit_feedback["reward"],
            bandit_feedback["position"],
            bandit_feedback["context"],
        ),
        total=bandit_feedback["n_rounds"],
    ):
        # Select action(s)
        if policy.policy_type == PolicyType.CONTEXT_FREE:
            selected_actions = policy.select_action()
        elif policy.policy_type == PolicyType.CONTEXTUAL:
            selected_actions = policy.select_action(context_.reshape(1, dim_context))
        else:
            raise RuntimeError("Unknown policy_type")

        # Check logging-policy match
        action_match = (action_ == selected_actions[position_])

        # Record realized reward
        if action_match:
            realized_rewards.append(reward_)   # click (0 or 1)
        else:
            realized_rewards.append(0)

        if action_match:
            if policy.policy_type == PolicyType.CONTEXT_FREE:
                policy.update_params(action=action_, reward=reward_)
            else:
                policy.update_params(
                    action=action_,
                    reward=reward_,
                    context=context_.reshape(1, dim_context),
                )

        selected_actions_list.append(selected_actions)

    action_dist = convert_to_action_dist(
        n_actions=int(bandit_feedback["action"].max()) + 1,
        selected_actions=np.array(selected_actions_list),
    )

    return action_dist, np.array(realized_rewards)