from obp.policy import BaseContextualPolicy
from dataclasses import dataclass
import numpy as np
from obp.dataset import OpenBanditDataset
from obp.ope import OffPolicyEvaluation, ReplayMethod as RM
import matplotlib.pyplot as plt
from utils_simulator import run_bandit_simulation
from pathlib import Path

import time 



@dataclass
class LinUCB(BaseContextualPolicy):
    alpha: float = 1.0
    def __post_init__(self):
        super().__post_init__()

        self.A_inv = np.stack([np.identity(self.dim) for _ in range(self.n_actions)], axis=0)
        self.b = np.zeros((self.n_actions, self.dim))

        self.A_inv_temp = np.copy(self.A_inv)
        self.b_temp = np.copy(self.b)

        self.name = f"LinUCB w alpha = {self.alpha}"


    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select a list of actions."""
        A_inv = self.A_inv
        theta = A_inv @ self.b[..., None]
        theta = theta.squeeze(-1)
        thetax = np.sum(theta * context, axis=1)

        Ax = A_inv @ context[..., None]
        Ax = Ax.squeeze(-1)
        xAx = np.sum(Ax * context, axis=1)

        p = thetax + self.alpha * np.sqrt(xAx)

        return np.argsort(p)[::-1][:self.len_list]

    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update policy parameters."""
        context = context.flatten()
        
        A_inv = self.A_inv[action]

        A_inv = A_inv - np.outer(A_inv @ context, context.T @ A_inv) / (1 + context.T @ (A_inv @ context))

        self.A_inv_temp[action] = A_inv
        self.b_temp[action] += reward*context

        self.n_trial += 1
        self.action_counts[action] += 1

        if self.n_trial % self.batch_size == 0:
            self.A_inv = np.copy(self.A_inv_temp)
            self.b = np.copy(self.b_temp)



@dataclass
class LinTS(BaseContextualPolicy):
    alpha: float = 1.0
    def __post_init__(self):
        super().__post_init__()
        self.B_inv = np.tile(np.identity(self.dim), (self.n_actions, 1, 1))
        self.mu = np.zeros((self.n_actions, self.dim))
        self.f = np.copy(self.mu)

        self.B_inv_temp = np.copy(self.B_inv)
        self.mu_temp = np.copy(self.mu)
        self.f_temp = np.copy(self.f)

        self.name = f"LinTS w alpha = {self.alpha}"

    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select a list of actions."""
        B_inv = self.B_inv_temp      # (n_actions, d, d)
        mu = self.mu_temp            # (n_actions, d)

        
        # trick for vectorized sampling. taken from https://stackoverflow.com/questions/69399035/is-there-a-way-of-batch-sampling-from-numpys-multivariate-normal-distribution-i
        # Batched Cholesky: returns (n_actions, d, d)
        L = np.linalg.cholesky(self.alpha**2 * B_inv)

        # Standard normals z ~ N(0, I)
        z = np.random.randn(self.n_actions, self.dim)  # (n_actions, d)

        # theta samples: mu + L @ z
        # einsum: for each action a: L[a] @ z[a]
        mu_tilde = mu + np.einsum("aij,aj->ai", L, z)  # (n_actions, d)

        # Expected reward for each action
        p = np.sum(mu_tilde * context, axis=1)

        # Top-k actions
        return np.argsort(p)[::-1][:self.len_list]
    
    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update policy parameters."""
        context = context.flatten()
        
        B_inv = self.B_inv_temp[action]
        Bx = B_inv @ context
        denom = 1.0 + context.T @ Bx  # scalar
        B_inv_new = B_inv - np.outer(Bx, Bx) / denom
        self.B_inv_temp[action] = B_inv_new

        self.f_temp[action] += reward * context
        self.mu_temp[action] = B_inv_new @ self.f_temp[action]

        self.n_trial += 1
        self.action_counts[action] += 1

        if self.n_trial % self.batch_size == 0:
            self.B_inv = np.copy(self.B_inv_temp)
            self.f = np.copy(self.f_temp)
            self.mu = np.copy(self.mu_temp)

@dataclass
class Random(BaseContextualPolicy):
    def select_action(self, context: np.ndarray) -> np.ndarray:
        """Select a list of actions."""
        return np.random.choice(np.arange(self.n_actions), self.len_list, replace=False)
    
    def update_params(self, action: int, reward: float, context: np.ndarray) -> None:
        """Update policy parameters."""
        return


class SimpleContextualEnv:
    """
    Simple environment for testing contextual bandit algorthims.

    - Context x ~ N(0, I_d)
    - Each action a has weight vector theta[a] ~ N(0, I_d)
    - Expected reward: sigmoid(theta[a]·x)
    - Real reward: Bernoulli(expected_reward)
    """

    def __init__(self, n_actions: int, dim_context: int, random_state: int = 12345):
        self.n_actions = n_actions
        self.dim_context = dim_context
        self.rng = np.random.RandomState(random_state)
        # true parameters for each action
        self.theta = self.rng.normal(size=(n_actions, dim_context))

    def step(self):
        """Sample one context and rewards for all actions."""
        x = self.rng.normal(size=(self.dim_context,))  
        logits = np.sum(self.theta * x, axis=1)                   
        probs = 1.0 / (1.0 + np.exp(-logits))       
        rewards = self.rng.binomial(1, probs)       
        return x, rewards, probs  # context, realized rewards, expected rewards


def run_policy_in_env(policy, env: SimpleContextualEnv, n_rounds: int = 10000):
    chosen_expected_rewards = []
    optimal_expected_rewards = []

    for t in range(n_rounds):
        context, all_rewards, expected_probs = env.step()

        a = policy.select_action(context)[0]
        chosen_expected_rewards.append(expected_probs[a])
        optimal_expected_rewards.append(expected_probs.max())       
        r = all_rewards[a]

        policy.update_params(action=a, reward=r, context=context)

    return np.array(chosen_expected_rewards), np.array(optimal_expected_rewards)

def synthetic_policy_sim(n_actions: int, dim_context: int, n_rounds: int, policies: list):
    min_regret = None
    best_policy = None

    for p in policies:
        env = SimpleContextualEnv(
        n_actions=n_actions,
        dim_context=dim_context,
        random_state=0,
        )

        p_rewards, p_oracle = run_policy_in_env(p, env, n_rounds)

        regret = np.cumsum(p_oracle - p_rewards)

        if min_regret is None:
            min_regret = regret
            best_policy = p
        else:
            if regret[-1] < min_regret[-1]:
                min_regret = regret
                best_policy = p
        
    return min_regret, best_policy



def synthetic_sim():
    print("Running Syntheric Data Simulation")
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]

    n_actions = 10
    dim_context = 5
    lin_ucbs = [LinUCB(dim=dim_context, n_actions=n_actions, alpha=alpha) for alpha in alphas]
    lin_tss = [LinTS(dim=dim_context, n_actions=n_actions, alpha=alpha) for alpha in alphas]
    random_policy = [Random(dim=dim_context, n_actions=n_actions)]

    n_rounds = 10000

    start = time.time()
    print("simulating LinUCB...")
    ucb_regret, best_ucb = synthetic_policy_sim(n_actions, dim_context, n_rounds, lin_ucbs)
    print("Done")
    end = time.time()
    print(f"Took {(end - start):.1f}s")

    start = time.time()
    print("simulating LinTS...")
    ts_regret, best_ts = synthetic_policy_sim(n_actions, dim_context, n_rounds, lin_tss)
    print("Done")
    end = time.time()
    print(f"Took {(end - start):.1f}s")

    random_regret, _ = synthetic_policy_sim(n_actions, dim_context, n_rounds, random_policy)

    plt.figure(figsize=(6, 4))
    plt.plot(ucb_regret, label=f"LinUCB w alpha={best_ucb.alpha}")
    plt.plot(ts_regret, label=f"LinTS w alpha={best_ts.alpha}", linestyle = "-.")
    plt.plot(random_regret, label="Ranodm", linestyle = "--")

    plt.xlabel("Round")
    plt.ylabel("Regret")
    plt.title("LinUCB vs LinTS regret in synthetic contextual environment")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/linucb_vs_lints_regret_random.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(ucb_regret, label=f"LinUCB w alpha={best_ucb.alpha}")
    plt.plot(ts_regret, label=f"LinTS w alpha={best_ts.alpha}", linestyle = "-.")

    plt.xlabel("Round")
    plt.ylabel("Regret")
    plt.title("LinUCB vs LinTS regret in synthetic contextual environment")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/linucb_vs_lints_regret.png")
    plt.close()



def real_policy_sim(bandit_feedback, policies, algo_name: str):
    print("Running Real Data Simulation")
    bar_vals = []     # relative estimated policy value
    ci_lowers = []    # amount below bar for error bar
    ci_uppers = []    # amount above bar for error bar
    alphas = []       # alpha for each policy

    behavior_value = bandit_feedback["reward"].mean()

    for p in policies:
        start = time.time()
        print(f"Started simulation for policy: {p.name}")

        action_dist, rewards = run_bandit_simulation(bandit_feedback, p)

        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback,
            ope_estimators=[RM()],
        )

        print(f"Estimating policy value of {p.name}")
        estimated_policy_value, estimated_interval = ope.summarize_off_policy_estimates(
            action_dist=action_dist
        )

        rel_val = float(
            estimated_policy_value["relative_estimated_policy_value"].iloc[0]
        )
        lower = float(estimated_interval["95.0% CI (lower)"].iloc[0])
        upper = float(estimated_interval["95.0% CI (upper)"].iloc[0])

        # convert CI bounds to *relative* scale
        rel_lower = lower / behavior_value
        rel_upper = upper / behavior_value

        bar_vals.append(rel_val)
        ci_lowers.append(rel_val - rel_lower)
        ci_uppers.append(rel_upper - rel_val)
        alphas.append(p.alpha)

        print(estimated_policy_value)
        print(estimated_interval)
        end = time.time()
        print(f"Done in {(end - start):.1f}s")

    return (
        np.array(alphas),
        np.array(bar_vals),
        np.array(ci_lowers),
        np.array(ci_uppers),
    )

        
def real_data_sim():
    # 1) Load logged data
    dataset = OpenBanditDataset(
        behavior_policy="random", 
        campaign="all",
        data_path="open_bandit_dataset"
    )
    
    bandit_feedback = dataset.obtain_batch_bandit_feedback()

    context_dim = bandit_feedback["context"].shape[1]
    n_actions = bandit_feedback["n_actions"]
    len_list = dataset.len_list
    random_state = 12345

    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]


    lin_ucbs = [LinUCB(dim=context_dim, n_actions=n_actions, len_list=len_list, random_state=random_state, alpha=alpha) for alpha in alphas]
    lin_tss = [LinTS(dim=context_dim, n_actions=n_actions, len_list=len_list, random_state=random_state, alpha=alpha) for alpha in alphas]



    alphas_ucb, vals_ucb, err_low_ucb, err_up_ucb = real_policy_sim(
        bandit_feedback, lin_ucbs, algo_name="LinUCB"
    )
    alphas_ts, vals_ts, err_low_ts, err_up_ts = real_policy_sim(
        bandit_feedback, lin_tss, algo_name="LinTS"
    )


    x = np.arange(len(alphas_ucb))
    width = 0.35  # width of each bar

    plt.figure(figsize=(10, 6))

    # LinUCB bars (left in each group)
    plt.bar(
        x - width / 2,
        vals_ucb,
        width,
        yerr=[err_low_ucb, err_up_ucb],
        capsize=5,
        label="LinUCB",
    )

    # LinTS bars (right in each group)
    plt.bar(
        x + width / 2,
        vals_ts,
        width,
        yerr=[err_low_ts, err_up_ts],
        capsize=5,
        label="LinTS",
    )

    plt.xticks(x, [str(a) for a in alphas_ucb], rotation=45)
    plt.xlabel("alpha")
    plt.ylabel("Relative Estimated Policy Value")
    plt.title("LinUCB vs LinTS on OpenBanditDataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/LinUCB_vs_LinTS_by_alpha_bernouli.png")
    plt.close()





if __name__ == "__main__":
    Path("plots").mkdir(exist_ok=True)
    synthetic_sim()
    real_data_sim()