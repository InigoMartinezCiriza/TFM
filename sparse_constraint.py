import tensorflow as tf
import numpy as np

# --------------------------
# Sparse Connections Constraint
# --------------------------
class SparseConstraint(tf.keras.constraints.Constraint):
    def __init__(self, prob_connection):
        """
        Constructs a constraint that applies a fixed binary mask to weights.
        The mask is generated randomly based on prob_connection when the layer
        is built (or first called) using the shape of the weights it receives.
        The mask remains fixed thereafter.

        Args:
            prob_connection (float): Probability of a weight being non-zero.
                                     Must be in (0, 1].
        """
        if not 0.0 < prob_connection <= 1.0:
            raise ValueError("prob_connection must be > 0 and <= 1.")
        self.prob_connection = prob_connection
        # Mask created lazily based on weight shape
        self.mask = None 

    def __call__(self, w):
        """
        Applies the sparsity mask to the weight tensor w.
        Generates the mask on the first call based on w's shape.
        """
        if self.mask is None:
            # Generate the mask once based on the specific weight tensor's shape
            mask_np = np.random.rand(*w.shape) < self.prob_connection
            self.mask = tf.constant(mask_np, dtype=w.dtype)

        # Ensure mask shape matches w's shape
        if self.mask.shape != w.shape:
             raise ValueError(
                 f"Internal error: Mask shape {self.mask.shape} "
                 f"does not match weight shape {w.shape}. "
                 "Mask should be generated based on weight shape."
             )

        # Element-wise multiplication with the fixed mask enforces sparsity
        return w * self.mask

    def get_config(self):
        # Return only the necessary configuration
        return {'prob_connection': self.prob_connection}