import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd # Added for plotting function dependency
import matplotlib.pyplot as plt # Added for plotting function dependency

# Action mapping
ACT_FIXATE = 0       # Penalized action
ACT_CHOOSE_LEFT = 1
ACT_CHOOSE_RIGHT = 2

# --- Observation vector indices (Updated) ---
OBS_A_LEFT_CUE = 0   # -1.0 if A is Left, +1.0 if A is Right (B is Left)
OBS_N_LEFT = 1       # Scaled amount offered on the Left side
OBS_N_RIGHT = 2      # Scaled amount offered on the Right side
OBS_DIM = 3          # Observation dimension is now 3

# --- Modified SimpleEconomicChoiceEnv with Random Sides ---
class EconomicChoiceEnv(gym.Env):
    """
    Simplified Gymnasium environment for a single-step Economic Choice Task.
    Juice A can be randomly assigned to the Left or Right side each trial.

    - Agent makes one choice: Left or Right.
    - Fixating (Action 0) results in a penalty.
    - Observation: [A_Left_Cue (-1/+1), Scaled_Amount_Left, Scaled_Amount_Right].
    - Episode ends immediately after the choice.
    - Stores trial_juice_LR, trial_offer_BA, and chosen_action for compatibility
      with helper measurement/plotting functions.
    """
    metadata = {'render_modes': [], 'render_fps': 1}

    def __init__(self,
                 A_to_B_ratio=2.2,
                 reward_B=100,
                 abort_penalty=-10.0,
                 max_offer_amount=10.0
                 ):
        super().__init__()

        self.A_to_B_ratio = A_to_B_ratio
        self.R_B = float(reward_B)
        self.R_A = float(A_to_B_ratio * self.R_B)
        self.R_ABORTED = float(abort_penalty)
        self.max_offer_amount = float(max_offer_amount)

        # --- Action Space ---
        self.action_space = spaces.Discrete(3) # 0: Fixate, 1: Left, 2: Right

        # --- Observation Space (Updated) ---
        # [A_Left_Cue (-1/+1), Scaled_N_Left (>=0), Scaled_N_Right (>=0)]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, np.inf, np.inf], dtype=np.float32), # Amounts can exceed max_offer_amount
            shape=(OBS_DIM,),
            dtype=np.float32
        )

        # --- Trial setup ---
        # Offers: (Amount A, Amount B) - quantities offered
        self.offer_sets = [(1, 0), (2, 1), (1, 1), (1, 2), (1, 3),
                           (1, 4), (1, 6), (1, 10), (0, 2)]
        # Juice side assignment possibilities: ('Juice Left', 'Juice Right')
        self.juice_types = [('A', 'B'), ('B', 'A')]
        self.rng = np.random.default_rng()

        # --- State variables ---
        # Base offer amounts for the trial
        self.trial_nA = 0
        self.trial_nB = 0
        # Amounts presented on Left/Right sides
        self.trial_nL = 0
        self.trial_nR = 0
        # Actual reward available for choosing Left/Right this trial
        self.reward_actual_L = 0.0
        self.reward_actual_R = 0.0

        # --- Compatibility variables ---
        self.trial_juice_LR = None       # ('A', 'B') or ('B', 'A')
        self.trial_offer_BA = None       # (nB, nA) - needed for psychometric plot
        self.chosen_action = -1          # Action taken in the step

    def _select_trial_conditions(self):
        """Sets up juice/offer amounts and side assignment for a new trial."""
        # 1. Select base amounts of A and B
        self.trial_nA, self.trial_nB = random.choice(self.offer_sets)

        # 2. Randomly assign sides for A and B
        self.trial_juice_LR = random.choice(self.juice_types)

        # 3. Determine amounts and rewards on Left/Right based on side assignment
        juiceL, juiceR = self.trial_juice_LR
        if juiceL == 'A': # A is Left, B is Right
            self.trial_nL = self.trial_nA
            self.trial_nR = self.trial_nB
            self.reward_actual_L = self.trial_nA * self.R_A
            self.reward_actual_R = self.trial_nB * self.R_B
        else: # B is Left, A is Right
            self.trial_nL = self.trial_nB
            self.trial_nR = self.trial_nA
            self.reward_actual_L = self.trial_nB * self.R_B
            self.reward_actual_R = self.trial_nA * self.R_A

        # 4. Store compatibility info (Offer pair B:A is independent of side)
        self.trial_offer_BA = (self.trial_nB, self.trial_nA)


    def _get_observation(self):
        """Constructs the 3D observation vector."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # State 0: A_Left_Cue
        juiceL, _ = self.trial_juice_LR
        obs[OBS_A_LEFT_CUE] = -1.0 if juiceL == 'A' else 1.0

        # States 1, 2: Scaled Amounts on Left and Right
        scale = self.max_offer_amount if self.max_offer_amount > 0 else 1.0
        scaled_nL = self.trial_nL / scale
        scaled_nR = self.trial_nR / scale

        obs[OBS_N_LEFT] = np.clip(scaled_nL, 0.0, None) # Clip low end
        obs[OBS_N_RIGHT] = np.clip(scaled_nR, 0.0, None) # Clip low end

        return obs

    def _get_info(self):
        """Returns auxiliary information about the current state."""
        info = {
            # Base trial setup
            "trial_nA": self.trial_nA,
            "trial_nB": self.trial_nB,
            "trial_offer_BA": self.trial_offer_BA, # (nB, nA)
            "trial_juice_LR": self.trial_juice_LR, # ('A','B') or ('B','A')
            # Side-specific info for this trial
            "nL": self.trial_nL,
            "nR": self.trial_nR,
            "reward_actual_L": self.reward_actual_L,
            "reward_actual_R": self.reward_actual_R,
            # Env params
            "A_to_B_ratio": self.A_to_B_ratio,
            # Compatibility info (action added in step)
            "chosen_action": self.chosen_action
        }
        return info

    def reset(self, seed=None, options=None):
        """Resets the environment for a new single-step trial."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            random.seed(seed)

        self._select_trial_conditions()
        self.chosen_action = -1 # Reset chosen action
        observation = self._get_observation()
        info = self._get_info() # Get base info after conditions are set
        info["reward"] = 0.0    # Add placeholders for step results
        info["action"] = None
        return observation, info

    def step(self, action):
        """Performs the single decision step."""
        if not self.action_space.contains(action):
             raise ValueError(f"Invalid action: {action}. Action must be in {self.action_space}")

        # Store chosen action for compatibility / info dict
        self.chosen_action = action

        terminated = True # Episode always ends after one step
        truncated = False # Never truncated
        reward = 0.0

        # --- Determine Reward based on action and actual side rewards ---
        if action == ACT_FIXATE:
            reward = self.R_ABORTED
        elif action == ACT_CHOOSE_LEFT:
            reward = self.reward_actual_L # Reward for whatever is on the Left
        elif action == ACT_CHOOSE_RIGHT:
            reward = self.reward_actual_R # Reward for whatever is on the Right

        # Next observation state doesn't matter as episode terminated
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        # --- Get info ---
        info = self._get_info() # Get base info reflecting the trial setup
        info["action"] = action # Add action taken
        info["reward"] = reward # Add reward received

        # Determine if the choice was 'correct' (higher actual value side) if not aborted
        if action == ACT_CHOOSE_LEFT:
             info["is_correct_choice"] = self.reward_actual_L >= self.reward_actual_R
        elif action == ACT_CHOOSE_RIGHT:
             info["is_correct_choice"] = self.reward_actual_R >= self.reward_actual_L
        else: # Aborted
             info["is_correct_choice"] = None

        return observation, reward, terminated, truncated, info

    def render(self):
        pass
    def close(self):
        pass