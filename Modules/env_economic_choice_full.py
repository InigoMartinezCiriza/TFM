import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pandas as pd # Added for helper function dependency
import matplotlib.pyplot as plt # Added for helper function dependency


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
EPOCH_CHOICE_HOLD = 'choice_hold' # Reintroduced
EPOCH_END = 'end'


class EconomicChoiceEnv(gym.Env):
    """
    Gymnasium environment for the Economic Choice Task with required Choice Hold.

    - Agent must fixate during FIXATION and OFFER_DELAY.
    - During GO_CHOICE, agent chooses Left or Right.
    - After choosing, agent must HOLD the chosen action (Left/Right)
      during CHOICE_HOLD epoch for a specified duration.
    - Final juice reward is given only after successful hold completion.
    - Breaking fixation or hold results in R_ABORTED.
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
                 reward_choice_hold=0.01,
                 duration_params=[1500, 1000, 2000, 2000, 750]
                 ):
        """
        Initializes the environment.

        Args:
            dt (int): Simulation time step (ms).
            A_to_B_ratio (float): Relative value A vs B.
            reward_B (float): Base reward for one drop of juice B.
            abort_penalty (float): Penalty for fixation/hold breaks or timeout.
            input_noise_sigma (float): Std dev of noise on numerical inputs.
            reward_fixation (float): Reward per step for correct fixation.
            reward_go_fixation (float): Penalty per step for fixating during GO.
            reward_choice_hold (float): Reward per step for correct holding.
            duration_params (list): Durations [fixation, delay_min, delay_max,
                                      go_timeout, choice_hold] in ms.
        """
        super().__init__()

        self.dt = dt
        self.dt_sec = dt / 1000.0
        self.A_to_B_ratio = A_to_B_ratio
        self.R_B = float(reward_B)
        self.R_A = float(A_to_B_ratio * self.R_B)
        self.R_ABORTED = float(abort_penalty)
        self.sigma = input_noise_sigma
        self.noise_scale = 1.0 / np.sqrt(self.dt_sec) if self.dt_sec > 0 else 1.0

        # Store reward parameters
        self.R_fix_step = float(reward_fixation)
        self.R_go_fix_step = float(reward_go_fixation)
        self.R_hold_step = float(reward_choice_hold) # Reintroduced

        # --- Action and Observation Spaces ---
        self.action_space = spaces.Discrete(3)
        # Observation space remains the same
        self.observation_space = spaces.Box(low=-1.1, high=2.1, shape=(OBS_DIM,), dtype=np.float32)

        # --- Timing ---
        if len(duration_params) != 5:
            raise ValueError("duration_params must have 5 elements: [fixation, delay_min, delay_max, go_timeout, choice_hold]")
        self._durations_ms = {
            'fixation':    duration_params[0],
            'delay_min':   duration_params[1],
            'delay_max':   duration_params[2],
            'go_timeout':  duration_params[3],
            'choice_hold': duration_params[4] # Reintroduced
        }
        self.t_fixation_steps = self._ms_to_steps(self._durations_ms['fixation'])
        self.t_choice_hold_steps = self._ms_to_steps(self._durations_ms['choice_hold']) # Reintroduced
        self.t_choice_timeout_steps = self._ms_to_steps(self._durations_ms['go_timeout'])

        # --- Trial setup ---
        self.juice_types = [('A', 'B'), ('B', 'A')]
        self.offer_sets = [(0, 1), (1, 2), (1, 1), (2, 1), (3, 1),
                           (4, 1), (6, 1), (10, 1), (2, 0)]
        self.rng = np.random.default_rng()

        # --- State variables ---
        self.current_step = 0
        self.trial_juice_LR = None
        self.trial_offer_BA = None
        self.trial_nL = 0
        self.trial_nR = 0
        self.trial_rL = 0.0
        self.trial_rR = 0.0
        self.epochs = {}
        self.current_epoch_name = EPOCH_END
        self.t_go_signal_step = -1
        self.t_choice_made_step = -1 # Step when L/R choice made INITIALLY
        self.chosen_action = -1    # Action chosen (1 or 2) that needs holding

    def _ms_to_steps(self, ms):
        """Converts milliseconds to simulation steps."""
        return max(1, int(np.round(ms / self.dt))) if ms > 0 else 0

    def _calculate_epochs(self, delay_ms):
        """Calculates initial epoch boundaries. Hold epoch is dynamic."""
        t_fix_end = self.t_fixation_steps
        t_delay_steps = self._ms_to_steps(delay_ms)
        t_go_signal = t_fix_end + t_delay_steps
        t_choice_end = t_go_signal + self.t_choice_timeout_steps # End of choice *window*

        # Max trial time = latest possible end of hold + buffer
        # Latest hold starts at t_choice_end - 1 step
        # Latest hold ends at (t_choice_end - 1) + t_choice_hold_steps
        t_max = t_choice_end + self.t_choice_hold_steps + self._ms_to_steps(100)

        self.epochs = {
            EPOCH_FIXATION:    (0, t_fix_end),
            EPOCH_OFFER_DELAY: (t_fix_end, t_go_signal),
            EPOCH_GO_CHOICE:   (t_go_signal, t_choice_end),
            # CHOICE_HOLD start/end are determined dynamically
            EPOCH_CHOICE_HOLD: (np.inf, np.inf), # Placeholder
            EPOCH_END:         (t_max, t_max + 1),
            'tmax_steps': t_max
        }
        self.t_go_signal_step = t_go_signal
        # print(f"Epochs calculated: Fix={self.epochs[EPOCH_FIXATION]}, Offer={self.epochs[EPOCH_OFFER_DELAY]}, Go={self.epochs[EPOCH_GO_CHOICE]}, MaxSteps={t_max}, HoldSteps={self.t_choice_hold_steps}")


    def _get_current_epoch(self, step):
        """Determines the current epoch, handling the dynamic CHOICE_HOLD."""
        # --- Check if choice has been made (dynamic hold period check) ---
        if self.t_choice_made_step >= 0:
            hold_start_step = self.t_choice_made_step + 1
            hold_end_step = hold_start_step + self.t_choice_hold_steps
            if hold_start_step <= step < hold_end_step:
                # print(f" Step {step}: In Hold. Start {hold_start_step}, End {hold_end_step}")
                return EPOCH_CHOICE_HOLD
            elif step >= hold_end_step:
                 # print(f" Step {step}: Past Hold. End {hold_end_step}")
                 # Should be transitioning to END state (handled in step logic)
                 return EPOCH_END

        # --- If no choice made yet, check fixed epochs ---
        if self.epochs[EPOCH_FIXATION][0] <= step < self.epochs[EPOCH_FIXATION][1]:
            return EPOCH_FIXATION
        elif self.epochs[EPOCH_OFFER_DELAY][0] <= step < self.epochs[EPOCH_OFFER_DELAY][1]:
            return EPOCH_OFFER_DELAY
        elif self.epochs[EPOCH_GO_CHOICE][0] <= step < self.epochs[EPOCH_GO_CHOICE][1]:
             # Ensure choice wasn't somehow made in the same step we check this
             if self.t_choice_made_step == -1:
                 return EPOCH_GO_CHOICE
             else: # Should transition to HOLD on next step
                 return EPOCH_GO_CHOICE # Or EPOCH_END if something is wrong? Safer maybe? Let's stick to GO for now.

        # Past GO window without choice, or other edge cases
        # print(f" Step {step}: Defaulting to END.")
        return EPOCH_END


    def _select_trial_conditions(self):
        """Sets up juice/offer conditions for a new trial."""
        self.trial_juice_LR = random.choice(self.juice_types)
        nB, nA = random.choice(self.offer_sets)
        self.trial_offer_BA = (nB, nA)

        juiceL, juiceR = self.trial_juice_LR
        if juiceL == 'A':
            self.trial_nL, self.trial_nR = nA, nB
            self.trial_rL, self.trial_rR = nA * self.R_A, nB * self.R_B
        else:
            self.trial_nL, self.trial_nR = nB, nA
            self.trial_rL, self.trial_rR = nB * self.R_B, nA * self.R_A

        delay_ms = self.rng.uniform(self._durations_ms['delay_min'],
                                    self._durations_ms['delay_max'])
        self._calculate_epochs(delay_ms)

    def _get_observation(self, current_epoch):
        """Constructs the 4D observation vector."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # OBS_FIX_CUE: Active only during Fixation and Offer Delay
        if current_epoch in [EPOCH_FIXATION, EPOCH_OFFER_DELAY]:
            obs[OBS_FIX_CUE] = 1.0

        # Offer Info (A_LEFT, N_LEFT, N_RIGHT): Active during Offer, Go, *and* Hold
        if current_epoch in [EPOCH_OFFER_DELAY, EPOCH_GO_CHOICE, EPOCH_CHOICE_HOLD]:
            juiceL, juiceR = self.trial_juice_LR
            obs[OBS_A_LEFT_CUE] = -1.0 if juiceL == 'A' else 1.0

            scaling_factor = 10.0 # Consistent scaling
            scaled_nL = self.trial_nL / scaling_factor
            scaled_nR = self.trial_nR / scaling_factor

            if self.sigma > 0:
                noise_L = self.rng.normal(scale=self.sigma) * self.noise_scale
                noise_R = self.rng.normal(scale=self.sigma) * self.noise_scale
                scaled_nL += noise_L
                scaled_nR += noise_R

            # Use same clipping as before, ensure A_LEFT cue is clipped too
            obs[OBS_N_LEFT] = np.clip(scaled_nL, 0.0, 1.1)
            obs[OBS_N_RIGHT] = np.clip(scaled_nR, 0.0, 1.1)
            obs[OBS_A_LEFT_CUE] = np.clip(obs[OBS_A_LEFT_CUE], -1.0, 1.0)

        # Zero observation in END state
        if current_epoch == EPOCH_END:
             obs = np.zeros(OBS_DIM, dtype=np.float32)

        # Final clip to ensure observation is within defined space bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs


    def reset(self, seed=None, options=None):
        """Resets the environment for a new trial."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            random.seed(seed)

        self.current_step = 0
        self._select_trial_conditions()
        # Reset state variables
        self.current_epoch_name = self._get_current_epoch(self.current_step)
        self.t_choice_made_step = -1
        self.chosen_action = -1

        observation = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        info["reward"] = 0.0
        info["action"] = None
        # print(f"Reset complete. Epoch: {self.current_epoch_name}, Obs: {observation}")
        return observation, info


    def step(self, action):
        """Advances the environment by one time step, including hold logic."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be in {self.action_space}")

        terminated = False
        truncated = False
        reward = 0.0
        prev_epoch = self.current_epoch_name
        # print(f"Step: {self.current_step}, Epoch: {prev_epoch}, Action: {action}, ChoiceMadeStep: {self.t_choice_made_step}") # Debug

        abort = False # Flag for immediate termination due to error

        # --- Reward/Penalty based on action in CURRENT step/epoch ---
        if prev_epoch == EPOCH_FIXATION or prev_epoch == EPOCH_OFFER_DELAY:
            if action != ACT_FIXATE:
                abort = True
                reward = self.R_ABORTED
                # print("Abort: Broke fixation.")
            else:
                reward = self.R_fix_step

        elif prev_epoch == EPOCH_GO_CHOICE:
            if action == ACT_FIXATE:
                reward = self.R_go_fix_step
            elif action in [ACT_CHOOSE_LEFT, ACT_CHOOSE_RIGHT]:
                # --- INITIAL CHOICE MADE ---
                # Only record the *first* time a choice is made
                if self.t_choice_made_step == -1:
                    self.t_choice_made_step = self.current_step
                    self.chosen_action = action
                    # Reward for *this first step* could be R_hold_step or 0
                    reward = self.R_hold_step # Give reward for first hold step
                    # print(f"Initial Choice: Action {action} at step {self.current_step}")
                else:
                    # Already made a choice, now holding (or should be)
                    # If action matches, it's like a hold step reward.
                    # If action mismatches, it should be handled by CHOICE_HOLD logic next step.
                    # This state is transient as epoch should change next step.
                    # Let's assume they continue holding correctly this step.
                     if action == self.chosen_action:
                         reward = self.R_hold_step
                     else:
                         # Made initial choice, then immediately switched in GO phase?
                         # Treat as breaking hold early.
                         abort = True
                         reward = self.R_ABORTED
                         # print("Abort: Switched action immediately after initial choice within GO.")


        elif prev_epoch == EPOCH_CHOICE_HOLD:
            if action == self.chosen_action:
                # Correctly holding
                reward = self.R_hold_step
            else:
                # Broke hold
                abort = True
                reward = self.R_ABORTED
                # print("Abort: Broke hold.")


        # If an abort occurred (fixation/hold break), set terminated and end epoch
        if abort:
            terminated = True
            self.current_epoch_name = EPOCH_END # Force epoch to END

        # --- Advance time and check state transitions ONLY if not already terminated ---
        if not terminated:
            self.current_step += 1
            # Determine the epoch we will be in *after* advancing the step
            next_epoch = self._get_current_epoch(self.current_step)
            # print(f"  Advanced step to {self.current_step}. Tentative next epoch: {next_epoch}") # Debug

            # --- Check for successful completion of CHOICE_HOLD ---
            # This happens if we *were* in HOLD, and advancing time takes us *past* the hold end
            hold_end_step = (self.t_choice_made_step + 1 + self.t_choice_hold_steps) if self.t_choice_made_step != -1 else -1
            if prev_epoch == EPOCH_CHOICE_HOLD and self.current_step >= hold_end_step:
                 # Successfully held fixation for the whole duration!
                 terminated = True
                 next_epoch = EPOCH_END # Ensure epoch reflects termination
                 # Assign final juice reward
                 if self.chosen_action == ACT_CHOOSE_LEFT:
                     reward = self.trial_rL
                 elif self.chosen_action == ACT_CHOOSE_RIGHT:
                     reward = self.trial_rR
                 else: # Should not happen if chosen_action was set correctly
                      reward = 0.0
                 # print(f"Success: Hold completed. Final Reward {reward}")


            # --- Check specifically for GO_CHOICE timeout ---
            # This happens if we were in GO, advanced time past the GO window,
            # *and* no choice was *ever* made (t_choice_made_step is still -1)
            go_start, go_end = self.epochs[EPOCH_GO_CHOICE]
            if prev_epoch == EPOCH_GO_CHOICE and self.current_step >= go_end and self.t_choice_made_step == -1:
                 reward = self.R_ABORTED
                 terminated = True
                 next_epoch = EPOCH_END
                 # print("Abort: GO_CHOICE timeout.")


            # Check for general truncation (overall time limit)
            elif self.current_step >= self.epochs['tmax_steps']:
                 truncated = True
                 reward = 0.0 # No specific reward/penalty for truncation
                 next_epoch = EPOCH_END
                 # print("Truncated: Reached max steps.")

            # Update the current epoch name based on time advancement and checks
            self.current_epoch_name = next_epoch

        # --- Get next observation and info ---
        observation = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        info["action"] = action # Action taken *leading* to this state
        info["reward"] = reward # Reward received for the *previous* state/action pair


        # Ensure terminated and truncated are mutually exclusive
        if terminated:
            truncated = False

        # print(f"  Step result: Obs={observation}, Rew={reward}, Term={terminated}, Trunc={truncated}, Epoch={self.current_epoch_name}") # Debug
        return observation, reward, terminated, truncated, info

    def _get_info(self):
        """Returns auxiliary information about the current state."""
        is_correct = None
        # Correctness is only relevant if a choice was initiated
        if self.chosen_action != -1:
             if self.chosen_action == ACT_CHOOSE_LEFT:
                 is_correct = self.trial_rL >= self.trial_rR
             elif self.chosen_action == ACT_CHOOSE_RIGHT:
                 is_correct = self.trial_rR >= self.trial_rL

        info = {
            "step": self.current_step,
            "epoch": self.current_epoch_name,
            "juice_LR": self.trial_juice_LR,
            "offer_BA": self.trial_offer_BA,
            "nL": self.trial_nL,
            "nR": self.trial_nR,
            "rL": self.trial_rL,
            "rR": self.trial_rR,
            "chosen_action": self.chosen_action,
            "choice_time_step": self.t_choice_made_step,
            "is_correct_choice": is_correct, # Based on initial choice
            "A_to_B_ratio": self.A_to_B_ratio,
            "rewards_cfg": {
                "fix_step": self.R_fix_step,
                "go_fix_step": self.R_go_fix_step,
                "hold_step": self.R_hold_step, # Reintroduced
                "abort": self.R_ABORTED
             },
             # "epoch_boundaries": self.epochs, # Optional debug info
             # "hold_duration_steps": self.t_choice_hold_steps # Optional debug info
        }
        return info

    def render(self):
        pass
    def close(self):
        pass