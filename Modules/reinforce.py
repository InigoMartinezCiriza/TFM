import tensorflow as tf
from helper_functions import discount_rewards
from helper_functions import pad_hidden_states
from helper_functions import collect_other_measurements
import numpy as np

# --------------------------
# Training using REINFORCE with baseline and regularization
# --------------------------
def train_agent(env, agent, num_episodes=500, gamma=0.99, print_interval=10, record_history = 1, l2_actor=1e-4, l2_critic=1e-4):
    """
    Trains the agent using REINFORCE with baseline on an environment.
    Processes episode sequences for RNN updates.
    Collects hidden states ("firing rates") only if networks use recurrent layers.

    Args:
        env: The potentially wrapped environment instance.
        agent: The ActorCriticAgent instance. Assumes agent.actor and agent.critic
               have a 'layer_type' attribute (e.g., 'GRU_modified', 'GRU_standard', 'Dense').
        num_episodes: Number of episodes to train for.
        gamma: Discount factor.
        print_interval: Interval for printing progress.
        record_history: Interval for which the variables are saved.
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
    other_measurements_history = []

    # Determine if networks are recurrent based on the stored layer_type
    actor_is_recurrent = hasattr(agent.actor, 'layer_type') and 'GRU' in agent.actor.layer_type
    critic_is_recurrent = hasattr(agent.critic, 'layer_type') and 'GRU' in agent.critic.layer_type

    lambda_actor = l2_actor
    lambda_critic = l2_critic

    # Determine the action dimension size once
    act_size = env.action_space.n

    # Containers to store hidden states across all episodes
    actor_states_all = [] if actor_is_recurrent else None
    critic_states_all = [] if critic_is_recurrent else None

    max_steps_actor = 0
    max_steps_critic = 0

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        states = []
        actions = []
        rewards = []
        actor_hidden_states_ep = []
        critic_hidden_states_ep = []
        
        # Store hidden states
        current_actor_hidden = None
        current_critic_hidden = None

        while not done:
            # Agent selects action
            action, _, actor_hidden_states_after_select = agent.select_action(
                state, actor_hidden_states=current_actor_hidden, training=True
            )
            
            # Store the latest actor hidden state if recurrent
            if actor_is_recurrent:
                current_actor_hidden = actor_hidden_states_after_select
                actor_hidden_states_ep.append(current_actor_hidden[0].numpy().flatten())

            # Environment steps
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store data for this step
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            # Get critic value and hidden state if recurrent 
            if critic_is_recurrent:
                 _, critic_hidden_state_after_eval = agent.evaluate_state(
                     state, critic_hidden_states=current_critic_hidden, training=False
                 )
                 current_critic_hidden = critic_hidden_state_after_eval
                 critic_hidden_states_ep.append(current_critic_hidden[0].numpy().flatten())

            state = next_state

        # --- Networks update ---
        returns = discount_rewards(rewards, gamma)
        
        # Convert collected data to tensors with appropriate shapes (batch_size = 1, sequence_length, feature_dim)
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
            # Pass the whole sequence to the actor. Output shape: (1, sequence_length, act_size)
            all_probs, _ = agent.actor(states_sequence_tensor, hidden_states=None, training=True)

            # Create one-hot encoding for actions taken. Shape: (1, sequence_length, act_size)
            actions_one_hot = tf.one_hot(actions_tensor, depth=act_size)

            # Select probabilities of the actions actually taken. Shape: (1, sequence_length)
            probs_taken_actions = tf.reduce_sum(all_probs * actions_one_hot, axis=-1)

            # Calculate log probabilities. Shape: (1, sequence_length)
            log_probs = tf.math.log(probs_taken_actions + 1e-10)

            # Get values from Critic for advantage calculation. Output shape: (1, sequence_length, 1)
            all_values, _ = agent.critic(states_sequence_tensor, hidden_states=None, training=True)

            # Shape: (1, sequence_length)
            values = tf.squeeze(all_values, axis=-1)

            # Calculate advantage A(s,a) = R_t - V(s_t). Shape: (1, sequence_length)
            advantage = returns_tensor - values

            # Calculate actor loss: - mean[ log_prob * advantage ]
            actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantage))

            # L2 Regularization for Actor
            l2_reg_actor = tf.add_n([tf.nn.l2_loss(v) for v in agent.actor.trainable_weights if 'kernel' in v.name or 'recurrent_kernel' in v.name])
            actor_loss += lambda_actor * l2_reg_actor

        actor_grads = tape_actor.gradient(actor_loss, agent.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, clip_norm=1.0)
        agent.actor_optimizer.apply_gradients(zip(actor_grads, agent.actor.trainable_variables))

        # --- Critic Update ---
        with tf.GradientTape() as tape_critic:
            # Pass the whole sequence to the critic again. Output shape: (1, sequence_length, 1)
            all_values, _ = agent.critic(states_sequence_tensor, hidden_states=None, training=True)

            # Shape: (1, sequence_length)
            values = tf.squeeze(all_values, axis=-1)

            # Calculate critic loss: mean[ (R_t - V(s_t))^2 ]. Shape: scalar
            critic_loss = tf.reduce_mean(tf.square(returns_tensor - values))

            # L2 Regularization for Critic
            l2_reg_critic = tf.add_n([tf.nn.l2_loss(v) for v in agent.critic.trainable_weights if 'kernel' in v.name or 'recurrent_kernel' in v.name])
            critic_loss += lambda_critic * l2_reg_critic

        critic_grads = tape_critic.gradient(critic_loss, agent.critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, clip_norm=1.0)
        agent.critic_optimizer.apply_gradients(zip(critic_grads, agent.critic.trainable_variables))

        # --- Recording ---

        # --- Firing Rate recording ---
        if actor_is_recurrent:
            actor_hidden_states_ep = np.array(actor_hidden_states_ep).T  # shape: units x steps
            actor_states_all.append(actor_hidden_states_ep)
            max_steps_actor = max(max_steps_actor, actor_hidden_states_ep.shape[1])

        if critic_is_recurrent:
            critic_hidden_states_ep = np.array(critic_hidden_states_ep).T
            critic_states_all.append(critic_hidden_states_ep)
            max_steps_critic = max(max_steps_critic, critic_hidden_states_ep.shape[1])

        # --- Other measurements recording ---
        last_reward = rewards[-1]
        measurement = collect_other_measurements(env, done, last_reward)
        other_measurements_history.append(measurement)

        # --- Reward and loss recording ---
        total_rewards_history.append(sum(rewards))
        actor_loss_history.append(actor_loss.numpy())
        critic_loss_history.append(critic_loss.numpy())

        if episode % print_interval == 0:
            print(f"Episode {episode}\tTotal Reward: {sum(rewards):.2f}\tState {states[-1]}\t"
                  f"Actor Loss: {actor_loss.numpy():.4f}\tCritic Loss: {critic_loss.numpy():.4f}")
            
        actor_states_tensor = None
        if actor_is_recurrent:
            units_actor = actor_states_all[0].shape[0]
            actor_states_tensor = np.array([pad_hidden_states(s, max_steps_actor, units_actor) for s in actor_states_all])
            actor_states_tensor = np.transpose(actor_states_tensor, (1, 2, 0))

        critic_states_tensor = None
        if critic_is_recurrent:
            units_critic = critic_states_all[0].shape[0]
            critic_states_tensor = np.array([pad_hidden_states(s, max_steps_critic, units_critic) for s in critic_states_all])
            critic_states_tensor = np.transpose(critic_states_tensor, (1, 2, 0))

    return total_rewards_history, actor_loss_history, critic_loss_history, \
           actor_states_tensor, critic_states_tensor, other_measurements_history