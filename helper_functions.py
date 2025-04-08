import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
def plot_firing_rates(actor_rates_per_layer, critic_rates_per_layer):
    """
    Plots the final firing rates of the actor and critic networks arranged in a grid where:
      - Columns represent the network (actor in the left column, critic in the right column)
      - Rows represent the layers (from layer 1 to max_layers)
      
    Parameters:
      actor_rates_per_layer: List of lists. Each element corresponds to one layer of the actor network.
                             Each sublist contains firing rate vectors (numpy arrays of shape (hidden_size,))
                             recorded for each episode.
                             
      critic_rates_per_layer: List of lists. Each element corresponds to one layer of the critic network.
                              Each sublist contains firing rate vectors (numpy arrays of shape (hidden_size,))
                              recorded for each episode.
    """
    num_actor_layers = len(actor_rates_per_layer)
    num_critic_layers = len(critic_rates_per_layer)
    max_layers = max(num_actor_layers, num_critic_layers)
    
    # Create a subplot grid with max_layers rows and 2 columns (actor, critic)
    fig, axes = plt.subplots(max_layers, 2, figsize=(8, 4 * max_layers))
    
    # If max_layers is 1, ensure axes is 2D
    if max_layers == 1:
        axes = np.array([axes])
    
    # Loop through each layer (row)
    for i in range(max_layers):
        # Actor column (left)
        ax_actor = axes[i, 0]
        if i < num_actor_layers:
            # Convert list of firing rate vectors for this layer into a 2D matrix: rows = neurons, columns = episodes.
            firing_matrix = np.array(actor_rates_per_layer[i]).T
            im = ax_actor.imshow(firing_matrix, aspect='auto', cmap='viridis')
            ax_actor.set_title(f"Actor Layer {i+1}")
            ax_actor.set_xlabel("Episode")
            ax_actor.set_ylabel("Neuron")
            fig.colorbar(im, ax=ax_actor)
        else:
            # Hide axis if no actor layer exists at this row.
            ax_actor.axis('off')
        
        # Critic column (right)
        ax_critic = axes[i, 1]
        if i < num_critic_layers:
            firing_matrix = np.array(critic_rates_per_layer[i]).T
            im = ax_critic.imshow(firing_matrix, aspect='auto', cmap='viridis')
            ax_critic.set_title(f"Critic Layer {i+1}")
            ax_critic.set_xlabel("Episode")
            ax_critic.set_ylabel("Neuron")
            fig.colorbar(im, ax=ax_critic)
        else:
            ax_critic.axis('off')
    
    plt.tight_layout()
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