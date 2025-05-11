import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd # Added for plotting function dependency
import matplotlib.pyplot as plt # Added for plotting function dependency

# Action mapping
ACT_FIXATE = 0
ACT_CHOOSE_LEFT = 1
ACT_CHOOSE_RIGHT = 2

# --- Observation vector indices (Updated) ---
OBS_FIX_CUE = 0      # 1.0 during fixation period, 0.0 otherwise
OBS_A_LEFT_CUE = 1   # -1.0 if A is Left, +1.0 if A is Right
OBS_N_LEFT = 2       # Scaled amount offered on the Left side
OBS_N_RIGHT = 3      # Scaled amount offered on the Right side
OBS_DIM = 4          # Observation dimension is now 4

# --- Epoch names (for info dictionary clarity) ---
EPOCH_FIXATION = 'fixation'
EPOCH_DECISION = 'decision'
EPOCH_END = 'end'      # Terminal state


class EconomicChoiceEnv(gym.Env):
    """
    Economic Choice Task environment with a mandatory fixation period.

    - Agent must fixate (Action 0) for a set duration.
    - Breaking fixation early results in a penalty.
    - After successful fixation, the agent has one step to choose Left or Right.
    - Fixating during the decision step is penalized.
    - Observation: [Fix_Cue, A_Left_Cue, Scaled_N_Left, Scaled_N_Right].
    - Juice A can be randomly assigned to the Left or Right side each trial.
    - Stores trial info for compatibility with helper functions.
    """
    metadata = {'render_modes': [], 'render_fps': 1}

    def __init__(self,
                 dt=100, # Time step in milliseconds
                 fixation_duration_ms=500, # Duration of mandatory fixation
                 A_to_B_ratio=2.2,
                 reward_B=100,
                 abort_penalty=-10.0,
                 reward_fix_step=0.0,
                 max_offer_amount=10.0
                 ):
        super().__init__()

        # --- Timing ---
        self.dt = dt
        self.dt_sec = dt / 1000.0
        self.fixation_duration_ms = fixation_duration_ms
        self.t_fixation_steps = self._ms_to_steps(fixation_duration_ms)
        if self.t_fixation_steps == 0:
             print("Warning: fixation_duration_ms is less than dt, setting fixation steps to 1.")
             self.t_fixation_steps = 1 # Ensure at least one fixation step if duration > 0

        # --- Rewards & Penalties ---
        self.A_to_B_ratio = A_to_B_ratio
        self.R_B = float(reward_B)
        self.R_A = float(A_to_B_ratio * self.R_B)
        self.R_ABORTED = float(abort_penalty)
        self.R_FIX_STEP = float(reward_fix_step)
        self.max_offer_amount = float(max_offer_amount)

        # --- Action Space ---
        self.action_space = spaces.Discrete(3) # 0: Fixate, 1: Left, 2: Right

        # --- Observation Space (Updated) ---
        # [Fix_Cue (0/1), A_Left_Cue (-1/+1), Scaled_N_Left (>=0), Scaled_N_Right (>=0)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, np.inf, np.inf], dtype=np.float32),
            shape=(OBS_DIM,),
            dtype=np.float32
        )

        # --- Trial setup ---
        self.offer_sets = [(1, 0), (2, 1), (1, 1), (1, 2), (1, 3),
                           (1, 4), (1, 6), (1, 10), (0, 2)]
        self.juice_types = [('A', 'B'), ('B', 'A')]
        self.rng = np.random.default_rng()

        # --- State variables ---
        self.current_step = 0
        self.current_epoch = EPOCH_END # Will be set in reset
        # Base offer amounts
        self.trial_nA = 0
        self.trial_nB = 0
        # Side-specific amounts
        self.trial_nL = 0
        self.trial_nR = 0
        # Side-specific rewards
        self.reward_actual_L = 0.0
        self.reward_actual_R = 0.0

        # --- Compatibility variables ---
        self.trial_juice_LR = None
        self.trial_offer_BA = None
        self.chosen_action = -1

    def _ms_to_steps(self, ms):
        """Converts milliseconds to simulation steps, minimum 1 if ms > 0."""
        if ms <= 0:
            return 0
        steps = int(np.round(ms / self.dt))
        return max(1, steps) # Ensure at least 1 step if ms > 0

    def _select_trial_conditions(self):
        """Sets up juice/offer amounts and side assignment for a new trial."""
        # Select base amounts
        self.trial_nA, self.trial_nB = random.choice(self.offer_sets)
        # Randomly assign sides
        self.trial_juice_LR = random.choice(self.juice_types)
        # Determine side-specific amounts and rewards
        juiceL, juiceR = self.trial_juice_LR
        if juiceL == 'A':
            self.trial_nL, self.trial_nR = self.trial_nA, self.trial_nB
            self.reward_actual_L, self.reward_actual_R = self.trial_nA * self.R_A, self.trial_nB * self.R_B
        else:
            self.trial_nL, self.trial_nR = self.trial_nB, self.trial_nA
            self.reward_actual_L, self.reward_actual_R = self.trial_nB * self.R_B, self.trial_nA * self.R_A
        # Store compatibility info
        self.trial_offer_BA = (self.trial_nB, self.trial_nA)

    def _get_observation(self):
        """Constructs the 4D observation vector based on the current state."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # State 0: Fixation Cue
        # Active (1.0) only during the fixation epoch
        obs[OBS_FIX_CUE] = 1.0 if self.current_epoch == EPOCH_FIXATION else 0.0

        # State 1: A_Left_Cue
        # This info is available throughout fixation and decision
        if self.trial_juice_LR: # Check if trial conditions are set
            juiceL, _ = self.trial_juice_LR
            obs[OBS_A_LEFT_CUE] = -1.0 if juiceL == 'A' else 1.0
        else:
             obs[OBS_A_LEFT_CUE] = 0.0 # Should not happen after reset

        # States 2, 3: Scaled Amounts on Left and Right
        # Available throughout fixation and decision
        scale = self.max_offer_amount if self.max_offer_amount > 0 else 1.0
        scaled_nL = self.trial_nL / scale
        scaled_nR = self.trial_nR / scale
        obs[OBS_N_LEFT] = np.clip(scaled_nL, 0.0, None)
        obs[OBS_N_RIGHT] = np.clip(scaled_nR, 0.0, None)

        return obs

    def _get_info(self):
        """Returns auxiliary information about the current state."""
        info = {
            # Timing/Epoch
            "step": self.current_step,
            "epoch": self.current_epoch,
            "fixation_steps_required": self.t_fixation_steps,
            # Base trial setup
            "trial_nA": self.trial_nA,
            "trial_nB": self.trial_nB,
            "trial_offer_BA": self.trial_offer_BA, # (nB, nA)
            "trial_juice_LR": self.trial_juice_LR, # ('A','B') or ('B','A')
            # Side-specific info
            "nL": self.trial_nL,
            "nR": self.trial_nR,
            "reward_actual_L": self.reward_actual_L,
            "reward_actual_R": self.reward_actual_R,
            # Env params
            "A_to_B_ratio": self.A_to_B_ratio,
            # Compatibility info (action added in step)
            "chosen_action": self.chosen_action,
             # Correct choice depends on when action is assessed
            "is_correct_choice": None # Set in step on decision
        }
        return info

    def reset(self, seed=None, options=None):
        """Resets the environment for a new trial with fixation."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            random.seed(seed)

        self._select_trial_conditions()
        self.current_step = 0
        self.current_epoch = EPOCH_FIXATION # Start in fixation
        self.chosen_action = -1 # Reset chosen action

        observation = self._get_observation()
        info = self._get_info()
        # Add initial step info placeholders
        info["reward"] = 0.0
        info["action"] = None
        return observation, info

    def step(self, action):
        """Advances the environment by one time step."""
        if not self.action_space.contains(action):
             raise ValueError(f"Invalid action: {action}. Action must be in {self.action_space}")

        terminated = False
        truncated = False # No truncation implemented here
        reward = 0.0
        prev_epoch = self.current_epoch # Store epoch before step update

        # --- Process action based on current epoch ---
        if self.current_epoch == EPOCH_FIXATION:
            if action == ACT_FIXATE:
                # Correct fixation, give step reward
                reward = self.R_FIX_STEP
                self.current_step += 1
                # Check if fixation period is now complete
                if self.current_step >= self.t_fixation_steps:
                    self.current_epoch = EPOCH_DECISION # Transition to decision phase
                    # No termination yet, agent needs to make choice in next step
                else:
                    # Stay in fixation
                    pass # current_epoch remains EPOCH_FIXATION
            else:
                # Broke fixation early
                reward = self.R_ABORTED
                terminated = True
                self.chosen_action = action # Store the incorrect action
                self.current_epoch = EPOCH_END # End the trial

        elif self.current_epoch == EPOCH_DECISION:
            # This is the single step where the choice must be made
            self.chosen_action = action # Store the final action
            terminated = True # Episode always ends after this step
            self.current_epoch = EPOCH_END # Mark as ended

            if action == ACT_CHOOSE_LEFT:
                reward = self.reward_actual_L
            elif action == ACT_CHOOSE_RIGHT:
                reward = self.reward_actual_R
            elif action == ACT_FIXATE:
                # Penalize fixation during the decision phase
                reward = self.R_ABORTED
            # Increment step counter conceptually for the final step
            self.current_step += 1


        elif self.current_epoch == EPOCH_END:
             # Should not happen if reset is called correctly, but handle defensively
             print("Warning: step() called after environment terminated.")
             terminated = True

        # --- Get next observation and info ---
        observation = self._get_observation() # Observation reflects the *new* state/epoch
        info = self._get_info() # Info reflects trial setup and final action if terminated
        info["action"] = action # Add action taken in this step
        info["reward"] = reward # Add reward received in this step

        # Determine correctness only if a choice was made in the decision epoch
        if prev_epoch == EPOCH_DECISION and action in [ACT_CHOOSE_LEFT, ACT_CHOOSE_RIGHT]:
            if action == ACT_CHOOSE_LEFT:
                info["is_correct_choice"] = self.reward_actual_L >= self.reward_actual_R
            else: # action == ACT_CHOOSE_RIGHT
                info["is_correct_choice"] = self.reward_actual_R >= self.reward_actual_L
        else:
            info["is_correct_choice"] = None # Not applicable or aborted


        return observation, reward, terminated, truncated, info

    def render(self):
        pass
    def close(self):
        pass