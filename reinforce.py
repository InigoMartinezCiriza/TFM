import tensorflow as tf
from helper_functions import discount_rewards
import numpy as np

# --------------------------
# Training using REINFORCE with baseline, regularization, and action delay
# --------------------------
def train_agent(env, agent, num_episodes=500, gamma=0.99, print_interval=10, l2_actor=1e-4, l2_critic=1e-4):
    """
    Trains the agent using REINFORCE with baseline on an environment potentially
    wrapped with action delay. Processes episode sequences for RNN updates.
    Collects hidden states ("firing rates") only if networks use recurrent layers.

    Args:
        env: The potentially wrapped environment instance.
        agent: The ActorCriticAgent instance. Assumes agent.actor and agent.critic
               have a 'layer_type' attribute (e.g., 'GRU_modified', 'GRU_standard', 'Dense').
        num_episodes: Number of episodes to train for.
        gamma: Discount factor.
        print_interval: Interval for printing progress.
        l2_actor: L2 regularization strength for the actor.
        l2_critic: L2 regularization strength for the critic.

    Returns:
        Tuple containing:
            total_rewards_history (list): Total reward per episode.
            actor_loss_history (list): Actor loss per episode.
            critic_loss_history (list): Critic loss per episode.
            actor_firing_rates (list of lists or None): Firing rates per layer for the actor,
                                                        or None if actor uses Dense layers.
            critic_firing_rates (list of lists or None): Firing rates per layer for the critic,
                                                         or None if critic uses Dense layers.
    """
    total_rewards_history = []
    actor_loss_history = []
    critic_loss_history = []

    actor_firing_rates = None
    critic_firing_rates = None

    # Determine if networks are recurrent based on the stored layer_type
    # (Assuming you fixed Actor/CriticModel to store self.layer_type)
    actor_is_recurrent = hasattr(agent.actor, 'layer_type') and 'GRU' in agent.actor.layer_type
    critic_is_recurrent = hasattr(agent.critic, 'layer_type') and 'GRU' in agent.critic.layer_type

    lambda_actor = l2_actor
    lambda_critic = l2_critic

    # Determine the action dimension size once
    act_size = env.action_space.n # Get action size from environment

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        states = []
        actions = []
        rewards = []

        # Variables to store the final hidden states if network is recurrent
        final_actor_hidden_for_log = None
        final_critic_hidden_for_log = None
        
        # --- Store hidden states during rollout if needed ---
        # (This part depends on whether you need step-by-step states or just final)
        # For simplicity, this version focuses on logging the *final* state 
        # after the last action selection, matching the previous logic. 
        # You could modify this to store states at each step if desired.
        current_actor_hidden = None
        current_critic_hidden = None

        while not done:
            # Agent selects action, potentially using/updating recurrent state
            action, _, actor_hidden_states_after_select = agent.select_action(
                state, actor_hidden_states=current_actor_hidden, training=True
            )
            
            # Store the latest actor hidden state if recurrent
            if actor_is_recurrent:
                current_actor_hidden = actor_hidden_states_after_select
                final_actor_hidden_for_log = actor_hidden_states_after_select # Keep track of the last one for logging

            # Environment steps
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store data for this step
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Get critic value and hidden state *if critic is recurrent* 
            # (Optional during rollout, mainly needed for logging final state here)
            if critic_is_recurrent:
                 # If you needed critic state during rollout (e.g., for actor advantage),
                 # you'd call evaluate_state here. We only need the last one for logging.
                 _, critic_hidden_state_after_eval = agent.evaluate_state(
                     state, critic_hidden_states=current_critic_hidden, training=False
                 )
                 current_critic_hidden = critic_hidden_state_after_eval
                 final_critic_hidden_for_log = critic_hidden_state_after_eval # Keep track of last one

            state = next_state
            
        # --- Firing Rate Logging (using the final recorded hidden states) ---
        if actor_is_recurrent and final_actor_hidden_for_log:
            final_actor_states = [h.numpy().squeeze(0) for h in final_actor_hidden_for_log] # Squeeze batch dim
            if actor_firing_rates is None:
                actor_firing_rates = [[] for _ in range(len(final_actor_states))]
            for i, rate in enumerate(final_actor_states):
                if rate.ndim == 0: rate = rate.reshape(1,)
                actor_firing_rates[i].append(rate)

        if critic_is_recurrent and final_critic_hidden_for_log:
            final_critic_states = [h.numpy().squeeze(0) for h in final_critic_hidden_for_log] # Squeeze batch dim
            if critic_firing_rates is None:
                critic_firing_rates = [[] for _ in range(len(final_critic_states))]
            for i, rate in enumerate(final_critic_states):
                 if rate.ndim == 0: rate = rate.reshape(1,)
                 critic_firing_rates[i].append(rate)

        # --- Networks update using full sequences ---
        returns = discount_rewards(rewards, gamma)
        
        # Convert collected data to tensors with appropriate shapes
        # Shape: (1, sequence_length, feature_dim)
        states_sequence_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        states_sequence_tensor = tf.expand_dims(states_sequence_tensor, axis=0)

        # Shape: (1, sequence_length)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        actions_tensor = tf.expand_dims(actions_tensor, axis=0)

        # Shape: (1, sequence_length)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        returns_tensor = tf.expand_dims(returns_tensor, axis=0)

        # --- Actor Update ---
        with tf.GradientTape() as tape_actor:
            # Pass the whole sequence to the actor
            # Output shape: (1, sequence_length, act_size)
            all_probs, _ = agent.actor(states_sequence_tensor, hidden_states=None, training=True)

            # Create one-hot encoding for actions taken
            # Shape: (1, sequence_length, act_size)
            actions_one_hot = tf.one_hot(actions_tensor, depth=act_size)

            # Select probabilities of the actions actually taken
            # Shape: (1, sequence_length)
            probs_taken_actions = tf.reduce_sum(all_probs * actions_one_hot, axis=-1)

            # Calculate log probabilities
            # Shape: (1, sequence_length)
            log_probs = tf.math.log(probs_taken_actions + 1e-10)

            # Get values from Critic for advantage calculation (run Critic forward pass here)
            # Output shape: (1, sequence_length, 1)
            all_values, _ = agent.critic(states_sequence_tensor, hidden_states=None, training=True) # Use training=True if critic has dropout etc.
            # Shape: (1, sequence_length)
            values = tf.squeeze(all_values, axis=-1)

            # Calculate advantage A(s,a) = R_t - V(s_t)
            # Shape: (1, sequence_length)
            advantage = returns_tensor - values

            # Calculate actor loss: - mean[ log_prob * advantage ]
            # Use stop_gradient on advantage for actor loss
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantage))

            # L2 Regularization for Actor
            l2_reg_actor = tf.add_n([tf.nn.l2_loss(v) for v in agent.actor.trainable_weights if 'kernel' in v.name or 'recurrent_kernel' in v.name])
            actor_loss += lambda_actor * l2_reg_actor

        actor_grads = tape_actor.gradient(actor_loss, agent.actor.trainable_variables)
        agent.actor_optimizer.apply_gradients(zip(actor_grads, agent.actor.trainable_variables))

        # --- Critic Update ---
        with tf.GradientTape() as tape_critic:
            # Pass the whole sequence to the critic again (necessary for gradient calc)
            # Output shape: (1, sequence_length, 1)
            all_values, _ = agent.critic(states_sequence_tensor, hidden_states=None, training=True)
            # Shape: (1, sequence_length)
            values = tf.squeeze(all_values, axis=-1)

            # Calculate critic loss: mean[ (R_t - V(s_t))^2 ]
            # Shape: scalar
            critic_loss = tf.reduce_mean(tf.square(returns_tensor - values))

            # L2 Regularization for Critic
            l2_reg_critic = tf.add_n([tf.nn.l2_loss(v) for v in agent.critic.trainable_weights if 'kernel' in v.name or 'recurrent_kernel' in v.name])
            critic_loss += lambda_critic * l2_reg_critic

        critic_grads = tape_critic.gradient(critic_loss, agent.critic.trainable_variables)
        agent.critic_optimizer.apply_gradients(zip(critic_grads, agent.critic.trainable_variables))

        # --- Logging ---
        episode_total_reward = sum(rewards)
        total_rewards_history.append(episode_total_reward)
        # Make sure to get scalar numpy values for logging
        actor_loss_history.append(actor_loss.numpy())
        critic_loss_history.append(critic_loss.numpy())

        if episode % print_interval == 0:
            print(f"Episode {episode}\tTotal Reward: {episode_total_reward:.2f}\t"
                  f"Actor Loss: {actor_loss.numpy():.4f}\tCritic Loss: {critic_loss.numpy():.4f}")

    return total_rewards_history, actor_loss_history, critic_loss_history, actor_firing_rates, critic_firing_rates