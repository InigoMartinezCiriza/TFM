import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Action mapping
ACT_FIXATE = 0
ACT_CHOOSE_LEFT = 1
ACT_CHOOSE_RIGHT = 2

# --- Observation vector indices ---
OBS_FIX_CUE = 0
OBS_A_LEFT_CUE = 1
OBS_N_LEFT = 2
OBS_N_RIGHT = 3
OBS_DIM = 4

# Epoch names
EPOCH_FIXATION = 'fixation'
EPOCH_OFFER_DELAY = 'offer_delay'
EPOCH_GO_CHOICE = 'go_choice'
EPOCH_CHOICE_HOLD = 'choice_hold'
EPOCH_END = 'end'


class EconomicChoiceEnv(gym.Env):
    """
    Gymnasium environment for the Economic Choice Task.
    Observation: [Fix Cue, A on Left, Num Left, Num Right]
    Rewards are configurable per epoch/action.
    Timeout during GO_CHOICE results in R_ABORTED.
    Fixating during GO_CHOICE gives reward_go_fixation (negative).
    """

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
            reward_B (float): Base reward for one drop of juice B (final large reward).
            abort_penalty (float): Penalty applied for critical errors (breaking fixation/hold, timeout).
            input_noise_sigma (float): Std dev of noise on numerical inputs.
            reward_fixation (float): Reward per step for correct fixation during FIXATION/OFFER_DELAY.
            reward_go_fixation (float): Reward/penalty per step for choosing FIXATE during GO_CHOICE.
            reward_choice_hold (float): Reward per step for correctly holding the chosen action during CHOICE_HOLD.
        """
        super().__init__()

        self.dt = dt
        self.dt_sec = dt / 1000.0
        self.A_to_B_ratio = A_to_B_ratio
        self.R_B = reward_B
        self.R_A = A_to_B_ratio * self.R_B
        self.R_ABORTED = abort_penalty
        self.sigma = input_noise_sigma
        self.noise_scale = 1.0 / np.sqrt(self.dt_sec) if self.dt_sec > 0 else 1.0

        # Store reward parameters
        self.R_fix_step = reward_fixation
        self.R_go_fix_step = reward_go_fixation
        self.R_hold_step = reward_choice_hold

        # --- Action and Observation Spaces ---
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=2.1, shape=(OBS_DIM,), dtype=np.float32)

        # --- Timing ---
        self._durations_ms = {
            'fixation':    duration_params[0],
            'delay_min':   duration_params[1],
            'delay_max':   duration_params[2],
            'go_timeout':  duration_params[3],
            'choice_hold': duration_params[4]
        }
        self.t_fixation_steps = self._ms_to_steps(self._durations_ms['fixation'])
        self.t_choice_hold_steps = self._ms_to_steps(self._durations_ms['choice_hold'])
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
        self.trial_rL = 0
        self.trial_rR = 0
        self.epochs = {}
        self.current_epoch_name = EPOCH_END
        self.t_go_signal_step = -1
        self.t_choice_made_step = -1
        self.chosen_action = -1

    def _ms_to_steps(self, ms):
        """Converts milliseconds to simulation steps."""
        return max(1, int(np.round(ms / self.dt)))

    def _calculate_epochs(self, delay_ms):
        """Calculates epoch boundaries in steps for the current trial."""
        t_fix_end = self.t_fixation_steps
        t_delay_steps = self._ms_to_steps(delay_ms)
        t_go_signal = t_fix_end + t_delay_steps
        t_choice_end = t_go_signal + self.t_choice_timeout_steps
        t_max = t_choice_end + self.t_choice_hold_steps + self._ms_to_steps(500)

        self.epochs = {
            EPOCH_FIXATION:    (0, t_fix_end),
            EPOCH_OFFER_DELAY: (t_fix_end, t_go_signal),
            EPOCH_GO_CHOICE:   (t_go_signal, t_choice_end),
            EPOCH_CHOICE_HOLD: (np.inf, np.inf),
            EPOCH_END:         (t_max, t_max + 1),
            'tmax_steps': t_max
        }
        self.t_go_signal_step = t_go_signal

    def _get_current_epoch(self, step):
        """Determines the current epoch name based on the step count."""
        if self.t_choice_made_step >= 0:
            hold_start = self.t_choice_made_step + 1
            hold_end = hold_start + self.t_choice_hold_steps
            if hold_start <= step < hold_end:
                return EPOCH_CHOICE_HOLD
            elif step >= hold_end:
                return EPOCH_END

        # Check fixed epochs
        if self.epochs[EPOCH_FIXATION][0] <= step < self.epochs[EPOCH_FIXATION][1]:
            return EPOCH_FIXATION
        elif self.epochs[EPOCH_OFFER_DELAY][0] <= step < self.epochs[EPOCH_OFFER_DELAY][1]:
            return EPOCH_OFFER_DELAY
        elif self.epochs[EPOCH_GO_CHOICE][0] <= step < self.epochs[EPOCH_GO_CHOICE][1]:
            # Check if choice already made (should not happen)
            if self.t_choice_made_step >= 0:
                 return EPOCH_END
            else:
                return EPOCH_GO_CHOICE
        else:
            return EPOCH_END

    def _select_trial_conditions(self):
        """Sets up juice/offer conditions for a new trial."""
        self.trial_juice_LR = random.choice(self.juice_types)
        offer_pairs_BA = [(0, 1), (1, 2), (1, 1), (2, 1), (3, 1),
                          (4, 1), (6, 1), (10, 1), (2, 0)]
        nB, nA = random.choice(offer_pairs_BA)
        self.trial_offer_BA = (nB, nA)

        juiceL, juiceR = self.trial_juice_LR
        if juiceL == 'A':
            self.trial_nL, self.trial_nR = nA, nB
            self.trial_rL, self.trial_rR = nA * self.R_A, nB * self.R_B
        else:
            self.trial_nL, self.trial_nR = nB, nA
            self.trial_rL, self.trial_rR = nB * self.R_B, nA * self.R_A

        delay_ms = self.rng.uniform(self._durations_ms['delay_min'], self._durations_ms['delay_max'])
        self._calculate_epochs(delay_ms)

    def _get_observation(self, current_epoch):
        """Constructs the 4D observation vector."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # State 0: Fixation Cue.
        # Fix cue is OFF (0.0) during GO_CHOICE and CHOICE_HOLD by default init
        if current_epoch in [EPOCH_FIXATION, EPOCH_OFFER_DELAY]:
            obs[OBS_FIX_CUE] = 1.0

        # States 1, 2, 3: Offer information (A_Left, N_Left, N_Right)
        # Active during Offer, Go, and Hold epochs
        if current_epoch in [EPOCH_OFFER_DELAY, EPOCH_GO_CHOICE, EPOCH_CHOICE_HOLD]:
            juiceL, juiceR = self.trial_juice_LR

            # State 1: A on Left Cue
            if juiceL == 'A':
                obs[OBS_A_LEFT_CUE] = -1.0
            else:
                obs[OBS_A_LEFT_CUE] = 1.0

            # States 2, 3: Scaled Amounts + Noise
            scaled_nL = self.trial_nL / 5.0
            scaled_nR = self.trial_nR / 5.0
            if self.sigma > 0:
                noise_L = self.rng.normal(scale=self.sigma) * self.noise_scale
                noise_R = self.rng.normal(scale=self.sigma) * self.noise_scale
                scaled_nL += noise_L
                scaled_nR += noise_R

            obs[OBS_N_LEFT] = np.clip(scaled_nL, 0.0, 2.1)
            obs[OBS_N_RIGHT] = np.clip(scaled_nR, 0.0, 2.1)

        return obs

    def reset(self, seed=None, options=None):
        """Resets the environment."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            random.seed(seed)

        self.current_step = 0
        self._select_trial_conditions()
        self.current_epoch_name = self._get_current_epoch(self.current_step)
        self.t_choice_made_step = -1
        self.chosen_action = -1

        observation = self._get_observation(self.current_epoch_name)
        info = self._get_info()
        return observation, info


    def step(self, action):
        """Advances the environment by one time step."""
        if not (0 <= action < self.action_space.n):
            raise ValueError(f"Invalid action: {action}. Action must be in [0, {self.action_space.n-1}]")

        terminated = False
        truncated = False
        # Default reward is 0
        reward = 0.0
        prev_epoch = self.current_epoch_name

        # --- Determine Reward/Penalty based on action in previous epoch ---
        # Flag for critical action errors needing immediate termination
        abort = False
        if prev_epoch == EPOCH_FIXATION or prev_epoch == EPOCH_OFFER_DELAY:
            if action != ACT_FIXATE:
                abort = True
                reward = self.R_ABORTED
            else:
                reward = self.R_fix_step

        elif prev_epoch == EPOCH_GO_CHOICE:
            if action == ACT_FIXATE:
                reward = self.R_go_fix_step
            elif action in [ACT_CHOOSE_LEFT, ACT_CHOOSE_RIGHT]:
                # Choice made. Record time and action. Assign step reward.
                self.t_choice_made_step = self.current_step
                self.chosen_action = action
                # No reward for making a choice
                reward = 0.0

        elif prev_epoch == EPOCH_CHOICE_HOLD:
            if action != self.chosen_action:
                # Failed to hold the chosen target correctly
                abort = True
                reward = self.R_ABORTED
            else:
                # Correctly holding the target
                reward = self.R_hold_step

        # Set terminated flag if an abort occurred due to ACTION error
        if abort:
            terminated = True
            # Force epoch to END if aborted
            self.current_epoch_name = EPOCH_END

        # --- If not terminated by action error, advance time and check for state-based termination/truncation ---
        if not terminated:
            self.current_step += 1
            # Determine the epoch we are entering after taking the step
            next_epoch = self._get_current_epoch(self.current_step)

            # --- Check specifically for GO_CHOICE timeout ---
            if prev_epoch == EPOCH_GO_CHOICE and next_epoch != EPOCH_GO_CHOICE and self.t_choice_made_step == -1:
                # We just transitioned out of GO_CHOICE, but no choice was made
                reward = self.R_ABORTED
                terminated = True
                next_epoch = EPOCH_END

            # --- Check for successful completion of CHOICE_HOLD ---
            elif prev_epoch == EPOCH_CHOICE_HOLD and next_epoch == EPOCH_END:
                 # Successfully held fixation. Assign final juice reward.
                 if self.chosen_action == ACT_CHOOSE_LEFT:
                     reward = self.trial_rL
                 elif self.chosen_action == ACT_CHOOSE_RIGHT:
                     reward = self.trial_rR
                 else:
                     reward = 0.0
                 terminated = True

            # --- Check for general truncation (overall time limit) ---
            elif not terminated and self.current_step >= self.epochs['tmax_steps']:
                 truncated = True
                 reward = 0.0
                 next_epoch = EPOCH_END

            # Update the current epoch name after all checks
            self.current_epoch_name = next_epoch

        # --- Get next observation and info ---
        # Observation uses the final 'current_epoch_name' for this step transition
        observation = self._get_observation(self.current_epoch_name)
        info = self._get_info()

        # Ensure terminated and truncated are mutually exclusive
        if terminated:
            truncated = False

        return observation, reward, terminated, truncated, info

    def _get_info(self):
        """Returns auxiliary information about the current state."""
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
            "is_correct_choice": self._was_correct() if self.t_choice_made_step >= 0 else None,
            "A_to_B_ratio": self.A_to_B_ratio,
            "rewards_cfg": {
                "fix_step": self.R_fix_step,
                "go_fix_step": self.R_go_fix_step,
                "hold_step": self.R_hold_step,
                "abort": self.R_ABORTED
             }
        }
        return info

    def _was_correct(self):
        """Checks if the choice made was for the higher value option."""
        # This logic remains the same
        if self.chosen_action == ACT_CHOOSE_LEFT:
            return self.trial_rL >= self.trial_rR
        elif self.chosen_action == ACT_CHOOSE_RIGHT:
            return self.trial_rR >= self.trial_rL
        return False