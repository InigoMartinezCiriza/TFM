import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
from modifiedGRU import ModifiedGRU
from sparse_constraint import SparseConstraint

# --------------------------
# Actor (Policy) Network
# --------------------------
@tf.keras.utils.register_keras_serializable()
class ActorModel(Model):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers=1,
                 prob_connection=1.0,
                 layer_type='GRU_standard',
                 alpha = 0.1,
                 **kwargs):
        """
        Actor network: Dense -> Hidden -> Dense.
        Args:
            input_size (int): Dimensionality of the observations (states).
            hidden_size (int): Number of units for the initial Dense layer and GRU layers.
            output_size (int): Number of actions.
            num_layers (int): Number of hidden layers.
            prob_connection (float): Connection probability for weights in hidden layer.
            layer_type (str): 'GRU_standard', 'GRU_modified' or 'Dense' units.
        """

        super(ActorModel, self).__init__(**kwargs)
        self.num_layers      = num_layers
        self.layer_type      = layer_type
        self.input_size      = input_size
        self.hidden_size     = hidden_size
        self.output_size     = output_size
        self.num_layers      = num_layers
        self.prob_connection = prob_connection
        self.layer_type      = layer_type
        self.alpha           = alpha

        # --- Helper function to create constraint instance ---
        def _create_new_constraint_if_sparse(prob):
            return SparseConstraint(prob) if prob < 1.0 else None

        # 1. Initial Dense Layer
        self.input_fc = layers.Dense(input_size, activation='relu', name='actor_input_dense')

        # 2. Hidden Layers
        self.hidden_layers = []
        for i in range(num_layers):
            layer_name_base = f'actor_{layer_type.lower().replace("_","")}_{i}'

            # --- Create constraint instances specifically for this hidden layer ---
            kernel_sparse_constraint_hidden = _create_new_constraint_if_sparse(prob_connection)
            recurrent_sparse_constraint_hidden = None
            if 'GRU' in layer_type:
                recurrent_sparse_constraint_hidden = _create_new_constraint_if_sparse(prob_connection)

            if layer_type == 'GRU_modified':
                self.hidden_layers.append(ModifiedGRU(hidden_size,
                                                      return_sequences=True, return_state=True,
                                                      kernel_constraint=kernel_sparse_constraint_hidden,
                                                      recurrent_constraint=recurrent_sparse_constraint_hidden,
                                                      name=layer_name_base,
                                                      alpha = alpha))
            elif layer_type == 'GRU_standard':
                self.hidden_layers.append(layers.GRU(hidden_size,
                                                     activation='tanh', recurrent_activation='sigmoid',
                                                     return_sequences=True, return_state=True,
                                                     kernel_constraint=kernel_sparse_constraint_hidden,
                                                     recurrent_constraint=recurrent_sparse_constraint_hidden,
                                                     name=layer_name_base))
            elif layer_type == 'Dense':
                self.hidden_layers.append(layers.Dense(hidden_size, activation='relu',
                                                       kernel_constraint=kernel_sparse_constraint_hidden,
                                                       name=layer_name_base))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        # 3. Final Dense Layer
        self.fc_out = layers.Dense(output_size, activation='softmax', name='actor_output_dense')

    def call(self, inputs, hidden_states=None, training=None):
        """
        Forward pass through Dense -> Hidden -> Dense.
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch, time_steps, input_size).
            hidden_states (list): List of initial hidden states for each GRU layer.
            training (bool): TensorFlow flag for training mode (e.g., for dropout).
        Returns:
            probs (tf.Tensor): Action probabilities (batch, output_size).
            new_states (list): List of final hidden states from each GRU layer.
        """
        processed_input = self.input_fc(inputs, training=training)
        output = processed_input
        new_states = []
        initial_states_for_call = [None] * self.num_layers

        if 'GRU' in self.layer_type and hidden_states is not None:
             initial_states_for_call = hidden_states

        for i, hidden_layer in enumerate(self.hidden_layers):
            current_initial_state = initial_states_for_call[i]
            if 'GRU' in self.layer_type:
                output, state = hidden_layer(output, initial_state=current_initial_state, training=training)
                new_states.append(state)
            elif self.layer_type == 'Dense':
                 output = hidden_layer(output, training=training)
            else:
                 raise TypeError(f"Layer type {self.layer_type} processing not handled in call.")

        probs = self.fc_out(output, training=training)
        return probs, new_states
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "num_layers": self.num_layers,
            "prob_connection": self.prob_connection,
            "layer_type": self.layer_type,
            "alpha": self.alpha
        })
        return config
    
    def get_hidden_dense(self, inputs):
        """
        Dado un tensor inputs de forma (batch, time, input_size),
        retorna la salida de la última capa oculta densa (batch, time, hidden_size),
        sin aplicar la capa fc_out.
        Sólo es válido si layer_type=='Dense'.
        """
        # Pasar por input_fc
        processed_input = self.input_fc(inputs, training=False)
        output = processed_input
        # Pasar por todas las hidden_layers (todas son Dense cuando layer_type=='Dense')
        for hidden_layer in self.hidden_layers:
            output = hidden_layer(output, training=False)
        return output

# --------------------------
# Critic (Value) Network
# --------------------------
@tf.keras.utils.register_keras_serializable()
class CriticModel(Model):
    def __init__(self,
                 actor_hidden_size,
                 act_size,
                 hidden_size,
                 num_layers=1,
                 prob_connection=1.0,
                 layer_type='GRU_standard',
                 alpha = 0.1,
                 **kwargs):
        """
        Critic network: Dense -> Hidden -> Dense.
        Args:
            input_size (int): Dimensionality of the observations.
            hidden_size (int): Number of units for the initial Dense layer and GRU layers.
            num_layers (int): Number of GRU layers.
            prob_connection (float): Connection probability for recurrent weights in GRU.
            layer_type (str): 'GRU_standard', 'GRU_modified' or 'Dense' units.
        """
        super(CriticModel, self).__init__(**kwargs)
        self.num_layers        = num_layers
        self.layer_type        = layer_type
        self.input_size        = actor_hidden_size + act_size
        self.actor_hidden_size = actor_hidden_size
        self.act_size          = act_size
        self.hidden_size       = hidden_size
        self.num_layers        = num_layers
        self.prob_connection   = prob_connection
        self.layer_type        = layer_type
        self.alpha             = alpha

        # --- Helper function to create constraint instance ---
        def _create_new_constraint_if_sparse(prob):
            return SparseConstraint(prob) if prob < 1.0 else None

        # 1. Initial Dense Layer
        self.input_fc = layers.Dense(self.input_size, activation='relu', name='critic_input_dense')

        # 2. Hidden Layers
        self.hidden_layers = []
        for i in range(num_layers):
            layer_name_base = f'critic_{layer_type.lower().replace("_","")}_{i}'

            # --- Create constraint instances specifically for this hidden layer ---
            kernel_sparse_constraint_hidden = _create_new_constraint_if_sparse(prob_connection)
            recurrent_sparse_constraint_hidden = None
            if 'GRU' in layer_type:
                 recurrent_sparse_constraint_hidden = _create_new_constraint_if_sparse(prob_connection)

            if layer_type == 'GRU_modified':
                 self.hidden_layers.append(ModifiedGRU(hidden_size,
                                                      return_sequences=True, return_state=True,
                                                      kernel_constraint=kernel_sparse_constraint_hidden,
                                                      recurrent_constraint=recurrent_sparse_constraint_hidden,
                                                      name=layer_name_base,
                                                      alpha = alpha))
            elif layer_type == 'GRU_standard':
                self.hidden_layers.append(layers.GRU(hidden_size,
                                                     activation='tanh', recurrent_activation='sigmoid',
                                                     return_sequences=True, return_state=True,
                                                     kernel_constraint=kernel_sparse_constraint_hidden,
                                                     recurrent_constraint=recurrent_sparse_constraint_hidden,
                                                     name=layer_name_base))
            elif layer_type == 'Dense':
                self.hidden_layers.append(layers.Dense(hidden_size, activation='relu',
                                                       kernel_constraint=kernel_sparse_constraint_hidden,
                                                       name=layer_name_base))
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

        # 3. Final Dense Layer
        self.fc_out = layers.Dense(1, name='critic_output_dense')

    def call(self, inputs, hidden_states=None, training=None):
        """
        Forward pass through Dense -> GRU -> Dense.
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch, time_steps, input_size).
            hidden_states (list): List of initial hidden states for each GRU layer.
            training (bool): TensorFlow flag for training mode.
        Returns:
            value (tf.Tensor): Estimated state value (batch, 1).
            new_states (list): List of final hidden states from each GRU layer.
        """
        processed_input = self.input_fc(inputs, training=training)
        output = processed_input
        new_states = []
        initial_states_for_call = [None] * self.num_layers
        if 'GRU' in self.layer_type and hidden_states is not None:
             initial_states_for_call = hidden_states

        for i, hidden_layer in enumerate(self.hidden_layers):
            current_initial_state = initial_states_for_call[i]
            if 'GRU' in self.layer_type:
                output, state = hidden_layer(output, initial_state=current_initial_state, training=training)
                new_states.append(state)
            elif self.layer_type == 'Dense':
                 output = hidden_layer(output, training=training)
            else:
                 raise TypeError(f"Layer type {self.layer_type} processing not handled in call.")

        value = self.fc_out(output, training=training)
        return value, new_states
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "actor_hidden_size": self.actor_hidden_size,
            "act_size": self.act_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "prob_connection": self.prob_connection,
            "layer_type": self.layer_type,
            "alpha": self.alpha
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Keras will pass everything that was in get_config() here
        return cls(
            config.pop("actor_hidden_size"),
            config.pop("act_size"),
            **config
        )

# --------------------------
# Actor-Critic Agent
# --------------------------
class ActorCriticAgent:
    def __init__(self,
                 obs_size,
                 act_size,
                 actor_hidden_size=128,
                 critic_hidden_size=128,
                 actor_layers=1,
                 critic_layers=1,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 noise_std=0.0,
                 actor_prob_connection=1.0,
                 critic_prob_connection=1.0,
                 layer_type='GRU_standard',
                 alpha = 0.1,
                 ):

        self.actor = ActorModel(
            input_size=obs_size, hidden_size=actor_hidden_size, output_size=act_size,
            num_layers=actor_layers, prob_connection=actor_prob_connection, layer_type=layer_type, alpha=alpha
        )

        actor_hid_for_critic = actor_hidden_size

        self.critic = CriticModel(
            actor_hidden_size=actor_hid_for_critic, act_size=act_size, hidden_size=critic_hidden_size, num_layers=critic_layers,
            prob_connection=critic_prob_connection, layer_type=layer_type, alpha=alpha
        )
        self.actor_optimizer = optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=critic_lr)
        self.noise_std = noise_std
        self.act_size = act_size

    def add_noise(self, state):
        if self.noise_std != 0.0:
            noise = np.random.normal(0, self.noise_std, size=state.shape).astype(state.dtype)
            return state + noise
        return state

    def select_action(self, state, actor_hidden_states=None, training=True):
        state_noisy = self.add_noise(state)
        state_tensor = tf.convert_to_tensor(state_noisy, dtype=tf.float32)
        state_tensor = tf.expand_dims(state_tensor, axis=0)
        state_tensor = tf.expand_dims(state_tensor, axis=1)
        probs, new_actor_hidden_states = self.actor(
            state_tensor, hidden_states=actor_hidden_states, training=training
        )
        probs_t = probs[:, 0, :]
        action = tf.random.categorical(tf.math.log(probs_t + 1e-10), num_samples=1)
        action = tf.squeeze(action, axis=-1)
        action_idx = tf.cast(action[0], dtype=tf.int32)
        action_one_hot = tf.one_hot(action_idx, depth=self.act_size)
        action_one_hot = tf.expand_dims(action_one_hot, axis=0)
        log_prob = tf.math.log(tf.reduce_sum(probs_t * action_one_hot, axis=1) + 1e-10)
        return int(action.numpy()[0]), log_prob[0], new_actor_hidden_states

    def evaluate_critic_step(self, r_t, prev_a, critic_hidden_states=None, training=False):
        # r_t: array float32 de shape (actor_hidden_size,)
        # prev_a: array float32 one-hot de shape (act_size,)
        if r_t is None:
            raise ValueError("Para actor Dense, use evaluate_critic_step con r_t calculado vía get_hidden_dense.")
        inp = np.concatenate([r_t, prev_a], axis=0)[None, None, :]  # (1,1,actor_hidden_size+act_size)
        inp = tf.convert_to_tensor(inp, dtype=tf.float32)
        value, new_critic_hidden_states = self.critic(inp, hidden_states=critic_hidden_states, training=training)
        scalar_value = value[0, 0, 0]
        return scalar_value, new_critic_hidden_states
        
    def save_full_models(self, filepath_prefix):
        actor_path = filepath_prefix + '_actor.keras'
        critic_path = filepath_prefix + '_critic.keras'
        self.actor.save(actor_path)
        self.critic.save(critic_path)
        print(f"Full models saved at {filepath_prefix}")

    def load_full_models(self, filepath_prefix):
        custom_objects = {
            'ModifiedGRU': ModifiedGRU,
            'SparseConstraint': SparseConstraint
        }

        actor_path = filepath_prefix + '_actor.keras'
        critic_path = filepath_prefix + '_critic.keras'

        self.actor = tf.keras.models.load_model(actor_path, compile=False, custom_objects=custom_objects)
        self.critic = tf.keras.models.load_model(critic_path, compile=False, custom_objects=custom_objects)

        print(f"Full models loaded from '{filepath_prefix}'")