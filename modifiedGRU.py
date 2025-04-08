import tensorflow as tf

# --------------------------
# Modified GRU unit
# --------------------------

class ModifiedGRUCell(tf.keras.layers.Layer):
    def __init__(self, units, alpha=0.1, kernel_constraint = None, recurrent_constraint = None, **kwargs):
        """
        A modified GRU cell that implements leaky, threshold-linear dynamics with dynamic time constants.
        Args:
            units (int): Number of units.
            alpha (float): A factor representing the discretized time step over the time constant.
            kernel_constraint: Constraint function applied to the `kernel` weights matrix
                (input transformations: W_in, Wl_in, Wg_in).
            recurrent_constraint: Constraint function applied to the `recurrent_kernel` weights matrix 
                (recurrent transformations: W_rec, Wl_rec, Wg_rec).
        """

        super(ModifiedGRUCell, self).__init__(**kwargs)
        self.units = units
        self.alpha = alpha
        self.state_size = units
        self.output_size = units
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Weights for the direct input and recurrent candidate term
        self.W_in = self.add_weight(shape=(input_dim, self.units),
                                    initializer='glorot_uniform',
                                    name='W_in',
                                    constraint=self.kernel_constraint)
        
        self.W_rec = self.add_weight(shape=(self.units, self.units),
                                     initializer='orthogonal',
                                     name='W_rec',
                                     constraint=self.recurrent_constraint)
        
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 name='b')
        
        # Weights for the gate l
        self.Wl_in = self.add_weight(shape=(input_dim, self.units),
                                     initializer='glorot_uniform',
                                     name='Wl_in',
                                     constraint=self.kernel_constraint)
        
        self.Wl_rec = self.add_weight(shape=(self.units, self.units),
                                      initializer='orthogonal',
                                      name='Wl_rec',
                                      constraint=self.recurrent_constraint)
        
        self.bl = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  name='bl')
        
        # Weights for the gate g
        self.Wg_in = self.add_weight(shape=(input_dim, self.units),
                                     initializer='glorot_uniform',
                                     name='Wg_in',
                                     constraint=self.kernel_constraint)
        
        self.Wg_rec = self.add_weight(shape=(self.units, self.units),
                                      initializer='orthogonal',
                                      name='Wg_rec',
                                      constraint=self.recurrent_constraint)
        
        self.bg = self.add_weight(shape=(self.units,),
                                  initializer='zeros',
                                  name='bg')
        
        super(ModifiedGRUCell, self).build(input_shape)

    def call(self, inputs, states):
        h_prev = states[0]

        # Compute the leak gate l that modulates the effective time constant
        l = tf.sigmoid(tf.matmul(h_prev, self.Wl_rec) + tf.matmul(inputs, self.Wl_in) + self.bl)

        # Compute the gate g that modulates the recurrent input
        g = tf.sigmoid(tf.matmul(h_prev, self.Wg_rec) + tf.matmul(inputs, self.Wg_in) + self.bg)

        # Compute the candidate update
        candidate = tf.matmul(g * h_prev, self.W_rec) + tf.matmul(inputs, self.W_in) + self.b

        # Update the state
        h_new = h_prev + self.alpha * l * (-h_prev + candidate)

        # Apply threshold-linear activation
        h_new = tf.maximum(0.0, h_new)

        return h_new, [h_new]
    
    def get_config(self):
        config = super(ModifiedGRUCell, self).get_config()
        config.update({
            'units': self.units,
            'alpha': self.alpha,
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': tf.keras.constraints.serialize(self.recurrent_constraint),
        })
        return config

class ModifiedGRU(tf.keras.layers.Layer):
    def __init__(self, units, return_sequences=False, return_state=False, alpha=0.1, kernel_constraint = None, recurrent_constraint = None, **kwargs):
        """
        Wrapper for the ModifiedGRUCell to mimic the API of tf.keras.layers.GRU.
        Args:
            units (int): Number of units.
            return_sequences (bool): Whether to return the full sequence or only the last output.
            return_state (bool): Whether to return the last state.
            alpha (float): Time-step over time constant factor.
            kernel_constraint: Constraint function applied to the `kernel` weights matrix
                (input transformations). Passed to ModifiedGRUCell.
            recurrent_constraint: Constraint function applied to the `recurrent_kernel`
                weights matrix (recurrent transformations). Passed to ModifiedGRUCell.
        """
        is_stateful = kwargs.pop('stateful', False)

        super(ModifiedGRU, self).__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.alpha = alpha
        self.cell = ModifiedGRUCell(units, alpha=alpha)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)
        self.stateful = is_stateful

        self.cell = ModifiedGRUCell(units, alpha=alpha,
                                    kernel_constraint=self.kernel_constraint,
                                    recurrent_constraint=self.recurrent_constraint,
                                    name=self.name + '_cell'
                                    )
        
        self.rnn_layer = tf.keras.layers.RNN(self.cell,
                                             return_sequences=self.return_sequences,
                                             return_state=self.return_state,
                                             stateful=self.stateful,
                                             name=self.name + '_rnn'
                                             )

    def build(self, input_shape):
        """Builds the underlying RNN layer."""
        if not self.rnn_layer.built:
             self.rnn_layer.build(input_shape)
        # Ensure this layer is marked as built
        super(ModifiedGRU, self).build(input_shape)


    def call(self, inputs, initial_state=None, training=None, mask=None):
        # Delegate the call to the internal RNN layer
        return self.rnn_layer(inputs,
                              initial_state=initial_state,
                              training=training,
                              mask=mask)
    
    @property
    def states(self):
        if self.stateful:
            return self.rnn_layer.states
        return None
    
    def get_initial_state(self, inputs):
         return self.rnn_layer.get_initial_state(inputs)

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        self.rnn_layer.reset_states(states)

    def get_config(self):
        config = super(ModifiedGRU, self).get_config()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'alpha': self.alpha,
            'stateful': self.stateful,
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'recurrent_constraint': tf.keras.constraints.serialize(self.recurrent_constraint),
        })
        # Remove cell config if present
        config.pop('cell', None)
        # Remove rnn_layer config if present
        config.pop('rnn_layer', None)
        return config