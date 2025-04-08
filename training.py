import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gymnasium as gym
from helper_functions import plot_firing_rates, plot_metrics 
from actor_critic import ActorCriticAgent
from delayed_cartpole import DelayedCartPoleEnv
from reinforce import train_agent

# --------------------------
# Main Function to Run Training with Delay
# --------------------------
if __name__ == '__main__':
    # --- Configuration ---
    config = {
        "env_name": "CartPole-v1",
        "t_delay": 5,
        "actor_hidden_size": 128,
        "critic_hidden_size": 128,
        "actor_layers": 1,
        "critic_layers": 1,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "num_episodes": 600,
        "gamma": 0.95,
        "noise_std": 0.01,
        "actor_prob_connection": 1.0,
        "critic_prob_connection": 1.0,
        "gru_type": "standard",
        "l2_actor": 1e-5,
        "l2_critic": 1e-5,
        "print_interval": 20
    }

    # --- Environment Setup ---
    base_env = gym.make(config["env_name"])
    # Wrap the base environment with the delay mechanism
    env = DelayedCartPoleEnv(base_env, t_delay=config["t_delay"])
    print(f"Using environment: {config['env_name']} with action delay t_delay={config['t_delay']}")

    # Get observation and action space sizes from the base environment
    obs_size = base_env.observation_space.shape[0]
    act_size = base_env.action_space.n
    print(f"Observation size: {obs_size}, Action size: {act_size}")


    # --- Agent Setup ---
    agent = ActorCriticAgent(
        obs_size, act_size,
        actor_hidden_size=config["actor_hidden_size"],
        critic_hidden_size=config["critic_hidden_size"],
        actor_layers=config["actor_layers"],
        critic_layers=config["critic_layers"],
        actor_lr=config["actor_lr"],
        critic_lr=config["critic_lr"],
        noise_std=config["noise_std"],
        actor_prob_connection=config["actor_prob_connection"],
        critic_prob_connection=config["critic_prob_connection"],
        gru_type=config["gru_type"]
    )

    # --- Training ---
    print("Starting training...")
    total_rewards_history, actor_loss_history, critic_loss_history, \
    actor_firing_rates, critic_firing_rates = train_agent(
        env=env,
        agent=agent,
        num_episodes=config["num_episodes"],
        gamma=config["gamma"],
        print_interval=config["print_interval"],
        l2_actor=config["l2_actor"],
        l2_critic=config["l2_critic"]
    )
    print("Training finished.")

    # --- Plotting Results ---
    print("Plotting metrics...")
    plot_metrics(total_rewards_history, actor_loss_history, critic_loss_history, window_1=10, window_2=50)
    print("Plotting firing rates...")
    plot_firing_rates(actor_firing_rates, critic_firing_rates)

    # Close the environment
    env.close()
    print("Done.")