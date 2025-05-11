import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from env_economic_choice import EconomicChoiceEnv as EconomicChoiceEnv
from env_economic_choice_1A3B import EconomicChoiceEnv as EconomicChoiceEnv_1A3B
from env_economic_choice_extreme import EconomicChoiceEnv as EconomicChoiceEnv_extreme
from env_economic_choice_simplest import EconomicChoiceEnv as EconomicChoiceEnv_simplest
from env_economic_choice_randomA import EconomicChoiceEnv as EconomicChoiceEnv_randomA
from env_economic_choice_fix import EconomicChoiceEnv as EconomicChoiceEnv_fix
from env_economic_choice_no_hold import EconomicChoiceEnv as EconomicChoiceEnv_nohold
from env_economic_choice_full import EconomicChoiceEnv as EconomicChoiceEnv_full
from env_partial_cartpole import CartPolePartialObservation

# --------------------------
# Helper function: Metrics plot
# --------------------------
def plot_metrics(total_rewards, actor_losses, critic_losses, window_1=10, window_2=25):
    """
    Plots the raw metrics and their rolling statistics over windows of window_1 and window_2 episodes.
    Each column corresponds to one metric:
      - Column 1: Total Reward
      - Column 2: Actor Loss
      - Column 3: Critic Loss
    Row 1: Raw metrics.
    Row 2: Rolling mean, median, and std over a window of window_1 episodes.
    Row 3: Rolling mean, median, and std over a window of window_2 episodes.
    """
    def plot_rolling(ax, data, window, label):
        series = pd.Series(data)
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_median = series.rolling(window=window, min_periods=1).median()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        ax.plot(episodes, rolling_mean, label="Mean")
        ax.plot(episodes, rolling_median, label="Median")
        ax.fill_between(episodes, rolling_mean - rolling_std, rolling_mean + rolling_std,
                        alpha=0.2, label="Std")
        ax.set_title(f"Rolling (window={window}) {label}")
        ax.set_xlabel("Episode")
        ax.legend()

    episodes = range(1, len(total_rewards) + 1)
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    
    axs[0, 0].plot(episodes, total_rewards, label="Raw")
    axs[0, 0].set_title("Raw Total Reward")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Total Reward")
    axs[0, 0].legend()
    
    axs[0, 1].plot(episodes, actor_losses, label="Raw", color="tab:orange")
    axs[0, 1].set_title("Raw Actor Loss")
    axs[0, 1].set_xlabel("Episode")
    axs[0, 1].set_ylabel("Actor Loss")
    axs[0, 1].legend()
    
    axs[0, 2].plot(episodes, critic_losses, label="Raw", color="tab:green")
    axs[0, 2].set_title("Raw Critic Loss")
    axs[0, 2].set_xlabel("Episode")
    axs[0, 2].set_ylabel("Critic Loss")
    axs[0, 2].legend()
    
    plot_rolling(axs[1, 0], total_rewards, window_1, "Total Reward")
    plot_rolling(axs[1, 1], actor_losses, window_1, "Actor Loss")
    plot_rolling(axs[1, 2], critic_losses, window_1, "Critic Loss")
    
    plot_rolling(axs[2, 0], total_rewards, window_2, "Total Reward")
    plot_rolling(axs[2, 1], actor_losses, window_2, "Actor Loss")
    plot_rolling(axs[2, 2], critic_losses, window_2, "Critic Loss")
    
    plt.tight_layout()
    plt.show()

# -------------------------
# Helper function: Firing rates plot
# --------------------------
def plot_firing_rates(actor_states_tensor, critic_states_tensor, network_name='Actor'):
    """
    Generates two plots:
    1. Mean firing rates as a function of the number of steps (averaged over units and valid episodes).
       Includes first and last episode firing rates.
    2. Mean firing rates as a function of units and episodes (averaged over valid steps only).

    Args:
        actor_states_tensor (np.ndarray): Tensor of shape (units, steps, episodes) for actor firing rates.
        critic_states_tensor (np.ndarray): Tensor of shape (units, steps, episodes) for critic firing rates.
        network_name (str): Name of the network to be displayed in the plot titles ('Actor' or 'Critic').
    """

    tensors = {'Actor': actor_states_tensor, 'Critic': critic_states_tensor}

    for name, tensor in tensors.items():
        if tensor is None:
            continue

        # Plot 1: Mean firing rates as a function of steps
        mean_over_steps = np.nanmean(tensor, axis=(0, 2))

        plt.figure(figsize=(10, 5))
        plt.plot(mean_over_steps, label='Mean firing rate')

        # Add first and last episodes
        first_episode = np.nanmean(tensor[:, :, 0], axis=0)
        last_episode = np.nanmean(tensor[:, :, -1], axis=0)

        plt.plot(first_episode, label='First episode', alpha=0.6)
        plt.plot(last_episode, label='Last episode', alpha=0.6)

        plt.xlabel('Steps')
        plt.ylabel('Mean firing rate')
        plt.title(f'{name} Mean firing rate over steps')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot 2: Mean firing rates as a function of units and episodes (valid steps only)
        mean_over_steps_units_episodes = np.nanmean(tensor, axis=1)

        plt.figure(figsize=(10, 6))
        plt.imshow(mean_over_steps_units_episodes, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Mean firing rate')
        plt.xlabel('Episodes')
        plt.ylabel('Units')
        plt.title(f'{name} Mean firing rate over units and episodes')
        plt.show()

# --------------------------
# Helper function: Discount reward
# --------------------------
def discount_rewards(rewards, gamma):
    """
    Computes discounted rewards.
    Args:
        rewards (list): List of rewards collected in an episode.
        gamma (float): Discount factor.
    Returns:
        np.array: Discounted rewards.
    """
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + gamma * cumulative
        discounted[i] = cumulative
    return discounted

# --------------------------
# Helper function: Padding for the hidden states
# --------------------------

def pad_hidden_states(hidden_states, max_steps, units):
    padded = np.full((units, max_steps), np.nan)
    length = hidden_states.shape[1]
    padded[:, :length] = hidden_states
    return padded

# --------------------------
# Helper function: Other measurements
# --------------------------

def collect_other_measurements(env, done, last_reward):
    """
    Collects environment-specific measurements at the end of an episode.

    Args:
        env: The environment instance.
        done (bool): Flag indicating if the episode ended.
        last_reward (float): The reward received in the final step.

    Returns:
        list or None: A list containing specific measurements for the target environment
                      if the episode ended successfully, otherwise None.
                      For EconomicChoiceEnv successful trials:
                      [juice_pair_LR, offer_pair_BA, chosen_juice_type]
                      e.g., [['A', 'B'], (3, 1), 'A'] means A was Left, B Right;
                            offer was 3 drops B vs 1 drop A; monkey chose A (Left).
    """
    env_instance = env.unwrapped if hasattr(env, 'unwrapped') else env

    # --- Partial CartPole Environment ---
    if isinstance(env_instance, CartPolePartialObservation):
        return None

    # --- Economic Choice Environment ---
    elif isinstance(env_instance, EconomicChoiceEnv) or \
         isinstance(env_instance, EconomicChoiceEnv_1A3B) or \
         isinstance(env_instance, EconomicChoiceEnv_extreme) or \
         isinstance(env_instance, EconomicChoiceEnv_simplest) or \
         isinstance(env_instance, EconomicChoiceEnv_randomA) or\
         isinstance(env_instance, EconomicChoiceEnv_fix) or \
         isinstance(env_instance, EconomicChoiceEnv_nohold) or \
         isinstance(env_instance, EconomicChoiceEnv_full):
        # Check for successful trial completion
        if done and last_reward >= env_instance.R_B:
            juice_pair = env_instance.trial_juice_LR
            offer_pair = env_instance.trial_offer_BA
            chosen_action = env_instance.chosen_action

            if chosen_action == 1:
                chosen_juice_type = juice_pair[0]
            elif chosen_action == 2:
                chosen_juice_type = juice_pair[1]
            else:
                chosen_juice_type = None

            if chosen_juice_type is not None and offer_pair is not None and juice_pair is not None:
                # Return juice assignment, offer amounts (B,A), and chosen type
                return [list(juice_pair), offer_pair, chosen_juice_type]
            else:
                return None
        else:
            # Episode did not end successfully with juice reward
            return None

    else:
        # Not the target environment
        return None

# --------------------------
# Helper function: Psycometric curves
# --------------------------

def plot_psychometric_curve(measurements_list, title="Psychometric Curve"):
    """
    Plots the percentage of times Juice B was chosen for each unique B:A offer pair.
    """

    successful_measurements = [m for m in measurements_list if m is not None]

    measurements_df = pd.DataFrame(
            successful_measurements,
            columns=['Juice_Pair_LR', 'Offer_Pair_BA', 'Chosen_Juice']
        )
    
    if measurements_df is None or measurements_df.empty:
        print("Cannot plot psychometric curve: No successful trial data provided.")
        return

    if 'Offer_Pair_BA' not in measurements_df.columns or 'Chosen_Juice' not in measurements_df.columns:
        print("Cannot plot psychometric curve: DataFrame missing required columns ('Offer_Pair_BA', 'Chosen_Juice').")
        return

    # Calculate choice counts per offer type
    choice_counts = measurements_df.groupby('Offer_Pair_BA')['Chosen_Juice'].value_counts().unstack(fill_value=0)

    # Ensure both 'A' and 'B' columns exist, even if one wasn't chosen for some offers
    if 'A' not in choice_counts.columns: choice_counts['A'] = 0
    if 'B' not in choice_counts.columns: choice_counts['B'] = 0

    # Calculate total trials per offer type
    choice_counts['Total'] = choice_counts['A'] + choice_counts['B']

    # Calculate percentage of B choices
    choice_counts['P(Choose B)'] = (choice_counts['B'] / choice_counts['Total']) * 100

    def sort_key(offer_pair_tuple):
        nB, nA = offer_pair_tuple
        # Treat B:0 offers as having a very high relative B value (or infinite)
        if nA == 0:
            return np.inf
        # Sort by the ratio of B drops to A drops
        return nB / nA

    # --- Prepare data for plotting ---
    plot_data = choice_counts.sort_index(key=lambda idx: idx.map(sort_key))

    # Create meaningful labels for the x-axis
    x_labels = [f"{nb}:{na}" for nb, na in plot_data.index]

    # --- Create the plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, plot_data['P(Choose B)'], marker='o', linestyle='-')

    plt.xlabel("Offer Type (# Drops B : # Drops A)")
    plt.ylabel("Percentage Choice (%) - Chose B")
    plt.title(title)
    plt.ylim([-5, 105])
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()