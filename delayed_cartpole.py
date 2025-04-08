import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math

class DelayedCartPoleEnv(gym.Wrapper):
    """
    A wrapper for the CartPole-v1 environment that introduces a delay
    between the agent choosing an action and the action being applied.

    During the delay period ('t_delay' steps), the environment physics are
    simulated *without applying any external force*, mimicking the natural
    evolution of the system while the agent "thinks".

    When step(action) is called:
    1. The chosen 'action' (0 or 1) is stored.
    2. The environment simulates 't_delay' internal physics steps with ZERO
       external force applied. Termination/truncation is checked at each internal step.
    3. If the episode hasn't terminated/truncated during the delay, the originally
       chosen 'action' (which translates to a force) is applied in a final physics step.
    4. The reward for the external action is fixed at +1.
    """
    def __init__(self, env: gym.Env, t_delay: int):
        """
        Initializes the delayed environment wrapper.

        Args:
            env: The base CartPole environment instance.
            t_delay: The number of time steps to delay the action (each step is tau seconds).
        """
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Discrete) or env.action_space.n != 2:
             raise ValueError("DelayedCartPoleEnv only supports CartPole-v1 with 2 actions.")
        if t_delay < 0:
            raise ValueError("t_delay must be non-negative.")

        self.t_delay = t_delay
        # The action and observation spaces remain unchanged.
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # Access physics parameters directly from the unwrapped environment.
        self.gravity = self.env.unwrapped.gravity
        self.masscart = self.env.unwrapped.masscart
        self.masspole = self.env.unwrapped.masspole
        self.total_mass = self.env.unwrapped.total_mass
        self.length = self.env.unwrapped.length  # actually half the pole's length
        self.polemass_length = self.env.unwrapped.polemass_length
        self.force_mag = self.env.unwrapped.force_mag
        self.tau = self.env.unwrapped.tau  # seconds between state updates

        # Get termination thresholds.
        self.theta_threshold_radians = self.env.unwrapped.theta_threshold_radians
        self.x_threshold = self.env.unwrapped.x_threshold

        # Use the base environment's max_episode_steps if defined.
        self._max_episode_steps = getattr(self.env.unwrapped, '_max_episode_steps', float('inf'))
        
        # Counter to track the number of external actions.
        self.external_step_count = 0

    def _simulate_step(self, state, force):
        """
        Performs one physics step using the CartPole dynamics.
        Matches the internal logic of CartPoleEnv.step().

        Args:
            state: Tuple (x, x_dot, theta, theta_dot).
            force: External force applied to the cart.

        Returns:
            Tuple: next_state (x, x_dot, theta, theta_dot)
        """
        x, x_dot, theta, theta_dot = state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Semi-implicit Euler integration
        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot

        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        """
        Executes the action with a delay, simulating zero force during the delay.
        Ensures that the reward for the external action is always +1.

        Args:
            action: The action chosen by the agent (0 or 1).

        Returns:
            (observation, reward, terminated, truncated, info) tuple.
        """
        # Increment the external action counter.
        self.external_step_count += 1

        if self.env.unwrapped.state is None:
            raise ValueError("Environment state is None. Call reset() before step().")

        current_state = self.env.unwrapped.state

        # --- Delay Phase (simulate with zero force) ---
        for _ in range(self.t_delay):
            # Simulate physics step with NO force.
            current_state = self._simulate_step(current_state, force=0.0)
            self.env.unwrapped.state = current_state

            x, _, theta, _ = current_state
            terminated = (x < -self.x_threshold or x > self.x_threshold or
                          theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)

            # Update the internal step counter.
            current_step = -1
            if hasattr(self.env.unwrapped, '_elapsed_steps'):
                self.env.unwrapped._elapsed_steps += 1
                current_step = self.env.unwrapped._elapsed_steps

            # Check for termination or truncation during delay.
            if terminated or current_step >= self._max_episode_steps:
                # Return with a fixed reward of +1.
                return np.array(current_state, dtype=np.float32), 1.0, terminated, current_step >= self._max_episode_steps, {}

        # --- Action Application Phase ---
        force = self.force_mag if action == 1 else -self.force_mag
        current_state = self._simulate_step(current_state, force=force)
        self.env.unwrapped.state = current_state

        x, _, theta, _ = current_state
        terminated = (x < -self.x_threshold or x > self.x_threshold or
                      theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)

        current_step = -1
        if hasattr(self.env.unwrapped, '_elapsed_steps'):
            self.env.unwrapped._elapsed_steps += 1
            current_step = self.env.unwrapped._elapsed_steps

        truncated = (current_step >= self._max_episode_steps)
        
        # --- Check if 500 external actions have been applied ---
        if self.external_step_count >= 500:
            truncated = True

        # Regardless of the internal delay steps, the reward is fixed at +1 per external action.
        reward = 1.0
        return np.array(current_state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, **kwargs):
        """Resets the underlying environment and the external action counter."""
        obs, info = self.env.reset(**kwargs)
        if self.env.unwrapped.state is None:
            self.env.unwrapped.state = tuple(obs)
        self.external_step_count = 0
        # Optionally reset the internal step counter if used.
        if hasattr(self.env.unwrapped, '_elapsed_steps'):
            self.env.unwrapped._elapsed_steps = 0
        return obs, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()