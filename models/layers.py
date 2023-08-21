import tensorflow as tf
import numpy as np

class Sampling(tf.keras.layers.Layer):
    """Custom Keras layer for sampling from a latent space.
    
    This layer takes as input the mean and log variance of a latent distribution
    and generates a sample using the reparameterization trick.
    
    Args:
        None
    
    Returns:
        Sampled latent vector.
    """

    def call(self, inputs):
        """Generate a latent sample using reparameterization trick.

        Args:
            inputs (tuple): A tuple containing mean and log variance of the latent distribution.

        Returns:
            Sampled latent vector.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class PositionEncoder(tf.keras.layers.Layer):
    """Positional encoding layer for temporal sequences.
    
    This layer encodes the temporal position of each element in a sequence using a combination of
    linear projection and embedding techniques.
    
    Args:
        temporal_dim (int): Number of elements in the sequence.
        projection_dim (int): Dimension of the projected and embedded positions.
    
    Returns:
        Encoded position information for input elements.
    """
    def __init__(self, temporal_dim, projection_dim):
        super(PositionEncoder, self).__init__()
        self.temporal_dim = temporal_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=temporal_dim, output_dim=projection_dim
        )

    def call(self, element):
        """Encode the temporal position of input elements.

        Args:
            element (tensor): Input sequence elements.

        Returns:
            Encoded position information for input elements.
        """
        positions = tf.range(start=0, limit=self.temporal_dim, delta=1)
        encoded = self.projection(element) + self.position_embedding(positions)
        return encoded

def positional_encoding(temporal_dim, projection_dim):
    """Generate sinusoidal positional encoding for temporal sequences.
    
    This function generates sinusoidal positional encodings based on the position and depth.
    
    Args:
        temporal_dim (int): Number of elements in the sequence.
        projection_dim (int): Dimension of the projected positions.
    
    Returns:
        Positional encoding tensor.
    """
    depth = projection_dim / 2

    positions = np.arange(temporal_dim)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

class AngularPositionEncoder(tf.keras.layers.Layer):
    """Angular positional encoding layer for temporal sequences.
    
    This layer combines a linear projection with sinusoidal positional encoding
    to provide angular information about element positions in a sequence.
    
    Args:
        temporal_dim (int): Number of elements in the sequence.
        projection_dim (int): Dimension of the projected positions.
    
    Returns:
        Encoded angular position information for input elements.
    """
    def __init__(self, temporal_dim, projection_dim):
        super(AngularPositionEncoder, self).__init__()
        self.temporal_dim = temporal_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)

    def call(self, element):
        """Encode angular position of input elements.

        Args:
            element (tensor): Input sequence elements.

        Returns:
            Encoded angular position information for input elements.
        """
        pos_encoding = positional_encoding(self.temporal_dim, self.projection.units)
        encoded = self.projection(element) + pos_encoding[tf.newaxis, ...]
        return encoded


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer Block layer.
    
    This class defines a single transformer block, which includes multi-head self-attention,
    feed-forward neural networks, layer normalization, and optional residual connections.
    
    Args:
        num_heads (int): Number of attention heads.
        d_model (int): Dimension of the model.
        droprate (float): Dropout rate applied to the output of each sub-layer.
        activation (str): Activation function to be used in feed-forward networks.
        res_connection (bool): Whether to use residual connections.
    
    Attributes:
        num_heads (int): Number of attention heads.
        d_model (int): Dimension of the model.
        droprate (float): Dropout rate applied to the output of each sub-layer.
        activation (function): Activation function to be used in feed-forward networks.
        res_connection (bool): Whether to use residual connections.
        mha (tf.keras.layers.MultiHeadAttention): Multi-head self-attention layer.
        ln_1 (tf.keras.layers.LayerNormalization): Layer normalization layer.
        dp_1 (tf.keras.layers.Dropout): Dropout layer for the first sub-layer.
        ln_2 (tf.keras.layers.LayerNormalization): Layer normalization layer.
        dp_2 (tf.keras.layers.Dropout): Dropout layer for the second sub-layer.
        fc_1 (tf.keras.layers.Dense): First feed-forward neural network.
        fc_2 (tf.keras.layers.Dense): Second feed-forward neural network.
    
    Methods:
        call(x, training=False): Perform a forward pass through the transformer block.
    """

    def __init__(self,
                 num_heads=4,
                 d_model=128,
                 droprate=0.1,
                 activation=tf.nn.gelu,
                 res_connection=True,
                 **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.droprate = droprate
        self.activation = activation
        self.res_connection = res_connection

    def build(self, input_shape):
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                                      key_dim=self.d_model//self.num_heads)
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dp_1 = tf.keras.layers.Dropout(self.droprate)
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dp_2 = tf.keras.layers.Dropout(self.droprate)
        self.fc_1 = tf.keras.layers.Dense(self.d_model*2, activation=self.activation)
        self.fc_2 = tf.keras.layers.Dense(self.d_model, activation=self.activation)

    def call(self, x, training=False):
        """Perform a forward pass through the transformer block.

        Args:
            x (tensor): Input tensor.
            training (bool): Whether the model is in training mode or not.

        Returns:
            Transformed output tensor.
        """
        x_h = self.ln_1(x)
        x_h = self.mha(x_h, x_h)
        x   = x + x_h if self.res_connection else x_h
        x_h = self.ln_2(x)
        x_h = self.fc_1(x_h)
        x_h = self.dp_1(x_h, training=training)
        x_h = self.fc_2(x_h)
        x_h = self.dp_2(x_h, training=training)
        x   = x + x_h if self.res_connection else x_h
        return x

class Wrapper(tf.keras.layers.Layer):
    def __init__(self, layer, flag=True, **kwargs):
        super(Wrapper, self).__init__(**kwargs)
        self.wrapper = tf.keras.layers.Bidirectional(layer) if flag else layer
    def call(self, x, initial_state=None):
        return self.wrapper(x) if initial_state is None\
               else self.wrapper(x, initial_state=initial_state)
        
class DifferenceLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Calculate differences between consecutive elements along the second dimension
        diffs = inputs[:, 1:, :] - inputs[:, :-1, :]
        diffs = tf.pad(diffs, paddings=[[0, 0], [1, 0], [0, 0]])
        
        return diffs
    
class ReverseDifferenceLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        # Cumulatively sum the differences to reconstruct the original data
        original_data = tf.cumsum(inputs, axis=1)
        return original_data


class GMM(tf.keras.layers.Layer):
    def __init__(self, num_clusters, projection_dim=None, **kwargs):
        super(GMM, self).__init__(**kwargs)
        self.num_clusters   = num_clusters
        self.projection_dim = projection_dim
    def build(self, input_shape):
        self.centers = self.add_weight(shape=(input_shape[-1], self.num_clusters),
                                        trainable=True,
                                        initializer='zeros',
                                        name='cluster_centers')
        self.sigma   = self.add_weight(shape=(input_shape[-1], self.num_clusters),
                                        trainable=False,
                                        initializer='ones',
                                        name='sigma') * 10.0
        self.projection = tf.keras.layers.Dense(units=self.projection_dim,
                                                input_shape=(input_shape[-1],))\
                         if self.projection_dim is not None else lambda x:x
        if self.projection_dim is not None:
            self.projection.build(input_shape=input_shape)
    def call(self, x):
        x       = self.projection(x)
        sigma_h = self.sigma**2.0 + 1e-7
        dist    = tf.exp(-tf.pow(x[..., tf.newaxis] - self.centers, 2.0)/sigma_h)
        probs   = tf.exp(tf.reduce_sum(dist, axis=1))/tf.reduce_sum(tf.exp(tf.reduce_sum(dist, axis=1)),axis=1,keepdims=True)
        return probs
    
    @property
    def means(self):
        return self.centers