import os
import sys
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import ObservationWrapper
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Used by helper_functions

# --- Custom Environment Wrapper ---

class CartPolePartialObservation(ObservationWrapper):
    """
    A wrapper for CartPole environments that only returns
    cart position (index 0) and pole angle (index 2) as observations.
    Updates the observation space accordingly.
    """
    def __init__(self, env):
        super().__init__(env)

        # --- Define the new observation space ---
        # Original observation space indices:
        # 0: Cart Position
        # 1: Cart Velocity (ignored)
        # 2: Pole Angle
        # 3: Pole Angular Velocity (ignored)

        original_low = self.env.observation_space.low # Access original space via self.env
        original_high = self.env.observation_space.high

        # Create new low/high arrays with only position and angle bounds
        new_low = np.array([original_low[0], original_low[2]], dtype=np.float32)
        new_high = np.array([original_high[0], original_high[2]], dtype=np.float32)

        # Set the new observation space for the wrapped environment
        self.observation_space = Box(low=new_low, high=new_high, dtype=np.float32)

    def observation(self, obs):
        """
        Filters the original observation to return only cart position and pole angle.
        """
        # obs is the original [pos, vel, angle, ang_vel] numpy array
        return obs[[0, 2]]