import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd # For helper function dependency
import matplotlib.pyplot as plt # For helper function dependency

# Action mapping
ACT_FIXATE = 0
ACT_CHOOSE_LEFT = 1
ACT_CHOOSE_RIGHT = 2

# --- Observation vector indices ---
OBS_FIX_CUE = 0
OBS_A_LEFT_CUE = 1 # -1.0 if A is Left, 1.0 if B is Left (A is Right)
OBS_N_LEFT = 2     # Scaled num drops left
OBS_N_RIGHT = 3    # Scaled num drops right
OBS_DIM = 4

# Epoch names
EPOCH_FIXATION = 'fixation'
EPOCH_OFFER_DELAY = 'offer_delay'
EPOCH_GO_CHOICE = 'go_choice'
# EPOCH_CHOICE_HOLD = 'choice_hold' # Removed
EPOCH_END = 'end'


class EconomicChoiceEnv(gym.Env):
    """
    Gymnasium environment for the Economic Choice Task with Fixation, Offer,
    and Go epochs, but NO CHOICE HOLD required.

    - Agent must fixate during FIXATION and OFFER_DELAY.
    - During GO_CHOICE, agent chooses Left or Right.
    - Choosing Left/Right during GO_CHOICE immediately ends the trial
      and delivers the corresponding reward.
    - Fixating during GO_CHOICE gives a small penalty per step.
    - Timeout during GO_CHOICE results in R_ABORTED.
    - Observation: [Fix Cue, A on Left, Num Left, Num Right]
    """
    metadata = {'render_modes': [], 'render_fps': 1}

    def __init__(self,
                 dt=10,
                 A_to_B_ratio=2.2,
                 reward_B=100,
                 abort_penalty=-0.1,
                 input_noise_sigma=0.0,
                 reward_fixation=0.01,
                 reward_go_fixation=-0.01,
                 duration_params=[1500, 1000, 2000, 2000]
                 ):
        """
        Initializes the environment.

        Args:
            dt (int): Simulation time step (ms).
            A_to_B_ratio (float): Relative value A vs B.
            reward_B (float): Base reward for one drop of juice B.
            abort_penalty (float): Penalty for fixation breaks or timeout.
            input_noise_sigma (float): Std dev of noise on numerical inputs.
            reward_fixation (float): Reward per step for correct fixation.
            reward_go_fixation (float): Penalty per step for fixating during GO.
            duration_params (list): Durations [fixation, delay_min, delay_max, go_timeout] in ms.
        """
        super().__init__()

        self.dt = dt
        self.dt_sec = dt / 1000.0
        self.A_to_B_ratio = A_to_B_ratio
        self.R_B = float(reward_B)
        self.R_A = float(A_to_B_ratio * self.R_B)
        self.R_ABORTED = float(abort_penalty)
        self.sigma = input_noise_sigma
        # Apply noise per sqrt(sec)
        self.noise_scale = 1.0 / np.sqrt(self.dt_sec) if self.dt_sec > 0 else 1.0

        # Store reward parameters
        self.R_fix_step = float(reward_fixation)
        self.R_go_fix_step = float(reward_go_fixation)
        # self.R_hold_step = float(reward_choice_hold) # Removed

        # --- Action and Observation Spaces ---
        self.action_space = spaces.Discrete(3)
        # Observation scaling remains the same as original
        self.observation_space = spaces.Box(low=-1.1, high=2.1, shape=(OBS_DIM,), dtype=np.float32)

        # --- Timing ---
        if len(duration_params) != 4:
            raise ValueError("duration_params must have 4 elements: [fixation, delay_min, delay_max, go_timeout]")
        self._durations_ms = {
            'fixation':    duration_params[0],
            'delay_min':   duration_params[1],
            'delay_max':   duration_params[2],
            'go_timeout':  duration_params[3],
            # 'choice_hold': duration_params[4] # Removed
        }
        self.t_fixation_steps = self._ms_to_steps(self._durations_ms['fixation'])
        # self.t_choice_hold_steps = self._ms_to_steps(self._durations_ms['choice_hold']) # Removed
        self.t_choice_timeout_steps = self._ms_to_steps(self._durations_ms['go_timeout'])

        # --- Trial setup ---
        self.juice_types = [('A', 'B'), ('B', 'A')] # Which juice is L/R
        # Offer sets are (nB, nA) pairs
        self.offer_sets = [(0, 1), (1, 2), (1, 1), (2, 1), (3, 1),
                           (4, 1), (6, 1), (10, 1), (2, 0)]
        self.rng = np.random.default_rng()

        # --- State variables ---
        self.current_step = 0
        self.trial_juice_LR = None # ('A', 'B') or ('B', 'A')
        self.trial_offer_BA = None # (nB, nA) chosen for trial
        self.trial_nL = 0          # n drops Left
        self.trial_nR = 0          # n drops Right
        self.trial_rL = 0.0        # reward value Left
        self.trial_rR = 0.0        # reward value Right
        self.epochs = {}           # Stores epoch start/end steps
        self.current_epoch_name = EPOCH_END
        self.t_go_signal_step = -1 # Step when GO signal appears
        self.t_choice_made_step = -1 # Step when L/R choice made in GO
        self.chosen_action = -1    # Action chosen (1 or 2)

    def _ms_to_steps(self, ms):
        """Converts milliseconds to simulation steps."""
        # Ensure at least 1 step for any non-zero duration
        return max(1, int(np.round(ms / self.dt))) if ms > 0 else 0

    def _calculate_epochs(self, delay_ms):
        """Calculates epoch boundaries in steps for the current trial."""
        t_fix_end = self.t_fixation_steps
        t_delay_steps = self._ms_to_steps(delay_ms)
        t_go_signal = t_fix_end + t_delay_steps
        # End of choice window (timeout)
        t_choice_end = t_go_signal + self.t_choice_timeout_steps
        # Max trial time is effectively the end of the choice window + buffer
        t_max = t_choice_end + self._ms_to_steps(100) # Small buffer after timeout

        self.epochs = {
            EPOCH_FIXATION:    (0, t_fix_end),
            EPOCH_OFFER_DELAY: (t_fix_end, t_go_signal),
            EPOCH_GO_CHOICE:   (t_go_signal, t_choice_end),
            # EPOCH_CHOICE_HOLD: No longer exists
            EPOCH_END:         (t_max, t_max + 1), # Point for truncation
            'tmax_steps': t_max
        }
        self.t_go_signal_step = t_go_signal
        # print(f"Epochs calculated: Fix={self.epochs[EPOCH_FIXATION]}, Offer={self.epochs[EPOCH_OFFER_DELAY]}, Go={self.epochs[EPOCH_GO_CHOICE]}, MaxSteps={t_max}")


    def _get_current_epoch(self, step):
        """Determines the current epoch name based on the step count.
           Assumes trial hasn't already ended by choice."""
        # --- Removed check for CHOICE_HOLD based on t_choice_made_step ---
        # if self.t_choice_made_step >= 0:
        #    # If choice was made, we should be in END state already via step() logic
        #    return EPOCH_END

        # Check fixed epochs based on step count progression
        if self.epochs[EPOCH_FIXATION][0] <= step < self.epochs[EPOCH_FIXATION][1]:
            return EPOCH_FIXATION
        elif self.epochs[EPOCH_OFFER_DELAY][0] <= step < self.epochs[EPOCH_OFFER_DELAY][1]:
            return EPOCH_OFFER_DELAY
        elif self.epochs[EPOCH_GO_CHOICE][0] <= step < self.epochs[EPOCH_GO_CHOICE][1]:
            # If we are within the GO window *and* no choice has been made yet
            if self.t_choice_made_step == -1:
                return EPOCH_GO_CHOICE
            else:
                # Should have been set to END by step() logic if choice was made
                # This case likely means something went wrong or we are querying
                # the epoch *after* the choice step but before returning from step().
                # Returning END is safest here.
                 return EPOCH_END
        else:
            # Past the GO_CHOICE window or other boundary conditions
            return EPOCH_END

    def _select_trial_conditions(self):
        """Sets up juice/offer conditions for a new trial."""
        self.trial_juice_LR = random.choice(self.juice_types)
        nB, nA = random.choice(self.offer_sets)
        self.trial_offer_BA = (nB, nA) # Store the canonical (B,A) offer

        juiceL, juiceR = self.trial_juice_LR
        if juiceL == 'A':
            self.trial_nL, self.trial_nR = nA, nB
            self.trial_rL, self.trial_rR = nA * self.R_A, nB * self.R_B
        else: # B is Left, A is Right
            self.trial_nL, self.trial_nR = nB, nA
            self.trial_rL, self.trial_rR = nB * self.R_B, nA * self.R_A

        # Random delay for the OFFER_DELAY epoch
        delay_ms = self.rng.uniform(self._durations_ms['delay_min'],
                                    self._durations_ms['delay_max'])
        self._calculate_epochs(delay_ms)

    def _get_observation(self, current_epoch):
        """Constructs the 4D observation vector based on the current epoch."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # OBS_FIX_CUE: Active during Fixation and Offer Delay
        if current_epoch in [EPOCH_FIXATION, EPOCH_OFFER_DELAY]:
            obs[OBS_FIX_CUE] = 1.0
        # If epoch is GO_CHOICE or END, Fix Cue is OFF (0.0)

        # OBS_A_LEFT_CUE, OBS_N_LEFT, OBS_N_RIGHT: Active during Offer Delay and Go Choice
        # We keep it active during GO so the agent remembers the offer
        if current_epoch in [EPOCH_OFFER_DELAY, EPOCH_GO_CHOICE]:
            juiceL, juiceR = self.trial_juice_LR

            # OBS_A_LEFT_CUE: -1 if A is Left, +1 if B is Left
            obs[OBS_A_LEFT_CUE] = -1.0 if juiceL == 'A' else 1.0

            # OBS_N_LEFT, OBS_N_RIGHT: Scaled amounts + optional noise
            # Scaling factor (e.g., divide by 5) - adjust if needed
            # Ensure scaling matches the observation space limits if strict bounds are desired
            scaling_factor = 10.0
            scaled_nL = self.trial_nL / scaling_factor
            scaled_nR = self.trial_nR / scaling_factor

            # Add noise if configured
            if self.sigma > 0:
                noise_L = self.rng.normal(scale=self.sigma) * self.noise_scale
                noise_R = self.rng.normal(scale=self.sigma) * self.noise_scale
                scaled_nL += noise_L
                scaled_nR += noise_R

            # Clip observation to be within reasonable bounds, e.g. [0, 2.1] as before
            # Adjust clipping based on scaling_factor and max expected offers
            obs[OBS_N_LEFT] = np.clip(scaled_nL, 0.0, 1.1)
            obs[OBS_N_RIGHT] = np.clip(scaled_nR, 0.0, 1.1)
            # Ensure the A_LEFT cue remains in its range
            obs[OBS_A_LEFT_CUE] = np.clip(obs[OBS_A_LEFT_CUE], -1.0, 1.0)


        # In EPOCH_END, the observation is typically all zeros or the last valid one.
        # Let's return zeros for simplicity after termination/truncation.
        if current_epoch == EPOCH_END:
             obs = np.zeros(OBS_DIM, dtype=np.float32)

        # Ensure observation fits the defined space bounds (important!)
        # The defined space is low=-1.1, high=2.1
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        return obs


    def reset(self, seed=None, options=None):
        """Resets the environment for a new trial."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            # Seed python's random too for random.choice
            random.seed(seed)

        self.current_step = 0
        self._select_trial_conditions()
        # Reset state variables
        self.current_epoch_name = self._get_current_epoch(self.current_step) # Should be FIXATION
        self.t_choice_made_step = -1
        self.chosen_action = -1

        observation = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        info["reward"] = 0.0 # Add initial reward info
        info["action"] = None

        # print(f"Reset complete. Epoch: {self.current_epoch_name}, Obs: {observation}")
        return observation, info


    def step(self, action):
        """Advances the environment by one time step."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be in {self.action_space}")

        terminated = False
        truncated = False
        reward = 0.0
        prev_epoch = self.current_epoch_name
        # print(f"Step: {self.current_step}, Epoch: {prev_epoch}, Action: {action}") # Debug


        # --- Determine Reward/Penalty based on action in CURRENT step/epoch ---
        abort = False # Flag for immediate termination due to error

        if prev_epoch == EPOCH_FIXATION or prev_epoch == EPOCH_OFFER_DELAY:
            if action != ACT_FIXATE:
                # Broke fixation
                abort = True
                reward = self.R_ABORTED
                # print("Abort: Broke fixation.")
            else:
                # Correct fixation, small positive reward
                reward = self.R_fix_step

        elif prev_epoch == EPOCH_GO_CHOICE:
            if action == ACT_FIXATE:
                # Choosing to fixate during Go is penalized but doesn't end trial
                reward = self.R_go_fix_step
            elif action in [ACT_CHOOSE_LEFT, ACT_CHOOSE_RIGHT]:
                # --- CHOICE MADE ---
                self.t_choice_made_step = self.current_step
                self.chosen_action = action
                terminated = True # End the trial immediately
                # Assign final reward based on choice
                if action == ACT_CHOOSE_LEFT:
                    reward = self.trial_rL
                else: # action == ACT_CHOOSE_RIGHT
                    reward = self.trial_rR
                # print(f"Choice made: Action {action}, Reward {reward}")
                # Force epoch to END state as trial is over
                self.current_epoch_name = EPOCH_END


        # --- (Removed EPOCH_CHOICE_HOLD logic) ---

        # If an abort occurred (fixation break), set terminated and end epoch
        if abort:
            terminated = True
            self.current_epoch_name = EPOCH_END

        # --- Advance time and check for state transitions ONLY if not already terminated ---
        if not terminated:
            self.current_step += 1
            # Determine the epoch we would enter *next* step if trial continues
            next_epoch = self._get_current_epoch(self.current_step)
            # print(f"  Advanced step to {self.current_step}. Tentative next epoch: {next_epoch}") # Debug

            # Check specifically for GO_CHOICE timeout
            # This happens if we were in GO_CHOICE, advanced time, and are now *past* the GO window boundary
            # *and* no choice was made in the *previous* step.
            go_start, go_end = self.epochs[EPOCH_GO_CHOICE]
            if prev_epoch == EPOCH_GO_CHOICE and self.current_step >= go_end and self.t_choice_made_step == -1:
                 # Timeout: Exceeded GO duration without making a valid choice
                 reward = self.R_ABORTED
                 terminated = True
                 next_epoch = EPOCH_END # Ensure epoch is set to END
                 # print("Abort: GO_CHOICE timeout.")


            # --- (Removed check for successful completion of CHOICE_HOLD) ---

            # Check for general truncation (overall time limit)
            # This catches runaway scenarios or unexpected delays
            elif self.current_step >= self.epochs['tmax_steps']:
                 truncated = True # Use truncated for hitting the max time limit
                 reward = 0.0     # No specific reward/penalty for truncation
                 next_epoch = EPOCH_END # Ensure epoch is set to END
                 # print("Truncated: Reached max steps.")

            # Update the current epoch name based on time advancement
            # (If termination occurred above, next_epoch was already set to END)
            self.current_epoch_name = next_epoch

        # --- Get next observation and info ---
        # Observation is based on the final current_epoch_name for this step transition
        observation = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        # Add runtime info to the dict for this step
        info["action"] = action
        info["reward"] = reward


        # Ensure terminated and truncated are mutually exclusive
        if terminated:
            truncated = False

        # print(f"  Step result: Obs={observation}, Rew={reward}, Term={terminated}, Trunc={truncated}, Epoch={self.current_epoch_name}") # Debug
        return observation, reward, terminated, truncated, info

    def _get_info(self):
        """Returns auxiliary information about the current state."""
        # Determine if the choice was correct (only if a choice was made)
        is_correct = None
        if self.chosen_action != -1:
             if self.chosen_action == ACT_CHOOSE_LEFT:
                 is_correct = self.trial_rL >= self.trial_rR
             elif self.chosen_action == ACT_CHOOSE_RIGHT:
                 is_correct = self.trial_rR >= self.trial_rL

        info = {
            "step": self.current_step,
            "epoch": self.current_epoch_name,
            "juice_LR": self.trial_juice_LR,
            "offer_BA": self.trial_offer_BA, # Canonical (B,A) offer
            "nL": self.trial_nL,
            "nR": self.trial_nR,
            "rL": self.trial_rL,
            "rR": self.trial_rR,
            "chosen_action": self.chosen_action, # -1 if no choice, 1 (L), 2 (R)
            "choice_time_step": self.t_choice_made_step, # -1 if no choice/timeout
            "is_correct_choice": is_correct,
            "A_to_B_ratio": self.A_to_B_ratio,
            "rewards_cfg": {
                "fix_step": self.R_fix_step,
                "go_fix_step": self.R_go_fix_step,
                # "hold_step": self.R_hold_step, # Removed
                "abort": self.R_ABORTED
             },
             # Include epoch timings for debugging/analysis if needed
             # "epoch_boundaries": self.epochs
        }
        return info

    # _was_correct method is integrated into _get_info now

    def render(self):
        """No rendering implemented."""
        pass

    def close(self):
        """Clean up any resources."""
        pass