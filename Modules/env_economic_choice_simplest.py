import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd # Added for plotting function dependency
import matplotlib.pyplot as plt # Added for plotting function dependency


# Action mapping
ACT_FIXATE = 0       # Penalized action
ACT_CHOOSE_LEFT = 1  # Choose A
ACT_CHOOSE_RIGHT = 2 # Choose B

# Observation vector indices (Simplified)
OBS_N_LEFT_A = 0    # Amount of A (always Left)
OBS_N_RIGHT_B = 1   # Amount of B (always Right)
OBS_DIM = 2

# --- Modified SimpleEconomicChoiceEnv ---
class EconomicChoiceEnv(gym.Env):
    """
    Simplified Gymnasium environment for a single-step Economic Choice Task.
    Modified to be compatible with the provided helper functions.

    - Agent makes one choice: Left (A) or Right (B).
    - Fixating (Action 0) results in a penalty.
    - Observation: [Amount Left (A), Amount Right (B)] scaled.
    - Episode ends immediately after the choice.
    - Juice A is always offered on the Left, Juice B on the Right.
    - Stores trial_juice_LR, trial_offer_BA, and chosen_action for compatibility.
    """
    metadata = {'render_modes': [], 'render_fps': 1}

    def __init__(self,
                 A_to_B_ratio=2.2,
                 reward_B=100,
                 abort_penalty=-0.1,
                 max_offer_amount=10.0
                 ):
        super().__init__()

        self.A_to_B_ratio = A_to_B_ratio
        self.R_B = float(reward_B)
        self.R_A = float(A_to_B_ratio * self.R_B)
        self.R_ABORTED = float(abort_penalty)
        self.max_offer_amount = float(max_offer_amount)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32
        )

        self.offer_sets = [(1, 0), (2, 1), (1, 1), (1, 2), (1, 3),
                           (1, 4), (1, 6), (1, 10), (0, 2)]
        self.rng = np.random.default_rng()

        # --- State variables ---
        self.trial_nA = 0
        self.trial_nB = 0
        self.trial_rL = 0.0
        self.trial_rR = 0.0

        # --- Compatibility variables ---
        self.trial_juice_LR = ('A', 'B') # Always A Left, B Right
        self.trial_offer_BA = None       # Will be (nB, nA)
        self.chosen_action = -1          # Action taken in the step

    def _select_trial_conditions(self):
        """Sets up juice/offer conditions for a new trial."""
        self.trial_nA, self.trial_nB = random.choice(self.offer_sets)
        self.trial_rL = self.trial_nA * self.R_A
        self.trial_rR = self.trial_nB * self.R_B
        # --- Store compatibility info ---
        self.trial_offer_BA = (self.trial_nB, self.trial_nA)

    def _get_observation(self):
        """Constructs the 2D observation vector [scaled_nA, scaled_nB]."""
        scale = self.max_offer_amount if self.max_offer_amount > 0 else 1.0
        scaled_nA = self.trial_nA / scale
        scaled_nB = self.trial_nB / scale
        obs = np.array([scaled_nA, scaled_nB], dtype=np.float32)
        return np.clip(obs, 0.0, None)

    def _get_info(self):
        """Returns auxiliary information about the current state."""
        info = {
            "nA": self.trial_nA,
            "nB": self.trial_nB,
            "rL": self.trial_rL,
            "rR": self.trial_rR,
            "A_to_B_ratio": self.A_to_B_ratio,
            # Include compatibility info for easy inspection
            "trial_juice_LR": self.trial_juice_LR,
            "trial_offer_BA": self.trial_offer_BA,
            # chosen_action is added during step
        }
        return info

    def reset(self, seed=None, options=None):
        """Resets the environment for a new single-step trial."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            random.seed(seed)

        self._select_trial_conditions()
        # Reset compatibility action variable
        self.chosen_action = -1
        observation = self._get_observation()
        info = self._get_info()
        info["reward"] = 0.0
        info["action"] = None
        return observation, info

    def step(self, action):
        """Performs the single decision step."""
        if not self.action_space.contains(action):
             raise ValueError(f"Invalid action: {action}. Action must be in {self.action_space}")

        # --- Store chosen action for compatibility ---
        self.chosen_action = action

        terminated = True
        truncated = False
        reward = 0.0

        if action == ACT_FIXATE:
            reward = self.R_ABORTED
        elif action == ACT_CHOOSE_LEFT:
            reward = self.trial_rL
        elif action == ACT_CHOOSE_RIGHT:
            reward = self.trial_rR

        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = self._get_info() # Get base info
        info["action"] = action # Add action taken to step info
        info["reward"] = reward # Add reward to step info

        # Determine if the choice was 'correct' (higher value) if not aborted
        if action == ACT_CHOOSE_LEFT:
             info["is_correct_choice"] = self.trial_rL >= self.trial_rR
        elif action == ACT_CHOOSE_RIGHT:
             info["is_correct_choice"] = self.trial_rR >= self.trial_rL
        else: # Aborted
             info["is_correct_choice"] = None

        return observation, reward, terminated, truncated, info

    def render(self):
        pass
    def close(self):
        pass