import tensorflow as tf
import numpy as np
from helper_functions import discount_rewards, pad_hidden_states, collect_other_measurements

# --------------------------
# Validation function for the trained agent
# --------------------------
def validate_agent(env, agent, num_episodes=500, gamma=0.99, print_interval=10, record_history=1):
    total_rewards_history = []
    other_measurements_history = []

    # Determine if actor is recurrent
    actor_is_recurrent = hasattr(agent.actor, 'layer_type') and 'GRU' in agent.actor.layer_type

    actor_states_all = [] if actor_is_recurrent else None
    max_steps_actor = 0

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        rewards = []
        actor_hidden_states_ep = []
        current_actor_hidden = None

        states = []

        while not done:
            action, _, actor_hidden_states_after_select = agent.select_action(
                state, actor_hidden_states=current_actor_hidden, training=False
            )

            if actor_is_recurrent:
                current_actor_hidden = actor_hidden_states_after_select
                actor_hidden_states_ep.append(current_actor_hidden[0].numpy().flatten())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            rewards.append(reward)
            state = next_state

        returns = discount_rewards(rewards, gamma)

        # Collect other measurements
        measurement = collect_other_measurements(env, done, rewards[-1])
        other_measurements_history.append(measurement)

        # Reward recording
        total_rewards_history.append(sum(rewards))

        if actor_is_recurrent:
            actor_hidden_states_ep = np.array(actor_hidden_states_ep).T  # shape: units x steps
            actor_states_all.append(actor_hidden_states_ep)
            max_steps_actor = max(max_steps_actor, actor_hidden_states_ep.shape[1])

        if episode % print_interval == 0:
            print(f"Validation Episode {episode}\tTotal Reward: {sum(rewards):.2f}\tFinal State: {states[-1]}")

    actor_states_tensor = None
    if actor_is_recurrent:
        units_actor = actor_states_all[0].shape[0]
        actor_states_tensor = np.array([pad_hidden_states(s, max_steps_actor, units_actor) for s in actor_states_all])
        actor_states_tensor = np.transpose(actor_states_tensor, (1, 2, 0))

    return total_rewards_history, actor_states_tensor, other_measurements_history