import tensorflow as tf
from .layers import Sampling, PositionEncoder, AngularPositionEncoder, TransformerBlock, DifferenceLayer, ReverseDifferenceLayer, Wrapper

def build_decoder_inputs(cfg):
    latent_inp = tf.keras.layers.Input((cfg.Model.latent), name='z_input')
    # direct_inp = tf.keras.layers.Input((cfg.Model.))
    
    initial_state = tf.keras.layers.Dense(units=2 * cfg.Model.LSTM.decoder_units[0],
                                          trainable=True if cfg.Model.LSTM.decoder_build_init_state else False,
                                          kernel_initializer='glorot_uniform' if cfg.Model.LSTM.decoder_build_init_state else 'zeros',
                                          use_bias=True if cfg.Model.LSTM.decoder_build_init_state else False,
                                          activation=cfg.Model.activation,
                                          name='dec_initial_state')(latent_inp)
    states = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, 1))(initial_state)
    init_state_model = tf.keras.Model(latent_inp, states, name='decoder_inputs_model')
    init_state_model.summary()
    return init_state_model 
    
def build_encoder(cfg):
    """Build the encoder network based on the provided configuration.
    
    This function constructs an encoder network based on the given configuration parameters.
    
    Args:
        cfg (Namespace): Configuration object containing model specifications.
    
    Returns:
        tf.keras.Model: The constructed encoder model.
    """
    inp = tf.keras.layers.Input((cfg.Model.temporal, cfg.Model.num_feat))
    x   = tf.keras.layers.Lambda(lambda x: x[:, cfg.Model.time_shift:, :], name='time_shift')(inp)
    x   = DifferenceLayer()(x) if cfg.Model.differential_input else x
    act = getattr(tf.nn, cfg.Model.activation)
    
    if cfg.Model.encoder_type == 'LSTM':
        units = cfg.Model.LSTM.encoder_units 
        return_seq = [True] * (len(units)-1) + [False]
        for (unit, rt_seq) in zip(units, return_seq):
            x = Wrapper(tf.keras.layers.LSTM(units=unit, 
                                     activation=act,
                                     dropout=cfg.Model.LSTM.decoder_dropout_rate,
                                     unroll=cfg.Model.LSTM.unroll, 
                                     return_sequences=rt_seq),
                                     cfg.Model.LSTM.encoder_bidirectional)(x)
                                     
    elif cfg.Model.encoder_type == 'Transformer':
        if cfg.Model.Transformer.position_encoder == 'angular' and cfg.Model.Transformer.encoder_encoding:
            x = AngularPositionEncoder()(x)
        elif cfg.Model.Transformer.position_encoder == 'embedding' and cfg.Model.Transformer.encoder_encoding:
            x = PositionEncoder()(x)
        else: 
            x = tf.keras.layers.Dense(units=cfg.Model.Transformer.d_model)(x)
            
        for i in range(cfg.Model.Transformer.encoder_blocks):
            x = TransformerBlock(num_heads=cfg.Model.Transformer.num_heads,
                                 d_model=cfg.Model.Transformer.d_model,
                                 droprate=cfg.Model.Transformer.encoder_droprate,
                                 activation=act,
                                 res_connection=cfg.Model.Transformer.res_connection)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    z_mean = tf.keras.layers.Dense(cfg.Model.latent, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(cfg.Model.latent, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(inp, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder


def build_decoder(cfg):
    """Build the decoder network based on the provided configuration.
    
    This function constructs a decoder network based on the given configuration parameters.
    
    Args:
        cfg (Namespace): Configuration object containing model specifications.
    
    Returns:
        tf.keras.Model: The constructed decoder model.
    """
    inp = tf.keras.layers.Input((cfg.Model.latent,))
    x = inp
    act = getattr(tf.nn, cfg.Model.activation)
    
    x = tf.keras.layers.Dense(cfg.Model.temporal * cfg.Model.num_feat, activation=act)(x)
    x = tf.keras.layers.Reshape((cfg.Model.temporal, cfg.Model.num_feat))(x)  
    
    if cfg.Model.decoder_type == 'LSTM':
        units = cfg.Model.LSTM.decoder_units 
        return_seq = [True] * (len(units))
        for (unit, rt_seq) in zip(units, return_seq):
            x = Wrapper(tf.keras.layers.LSTM(units=unit, 
                                     activation=act,
                                     dropout=cfg.Model.LSTM.decoder_dropout_rate,
                                     unroll=cfg.Model.LSTM.unroll, 
                                     return_sequences=rt_seq), 
                                     cfg.Model.LSTM.decoder_bidirectional)(x)


    elif cfg.Model.decoder_type == 'Transformer':
        if cfg.Model.Transformer.position_encoder == 'angular' and cfg.Model.Transformer.decoder_encoding:
            x = AngularPositionEncoder()(x)
        elif cfg.Model.Transformer.position_encoder == 'embedding' and cfg.Model.Transformer.decoder_encoding:
            x = PositionEncoder()(x)
        else: 
            x = tf.keras.layers.Dense(units=cfg.Model.Transformer.d_model)(x)
        for i in range(cfg.Model.Transformer.encoder_blocks):
            x = TransformerBlock(num_heads=cfg.Model.Transformer.num_heads,
                                 d_model=cfg.Model.Transformer.d_model,
                                 droprate=cfg.Model.Transformer.encoder_droprate,
                                 activation=act,
                                 res_connection=cfg.Model.Transformer.res_connection)(x)

    decoder_output = tf.keras.layers.Dense(cfg.Model.num_feat)(x)
    
    decoder_output = ReverseDifferenceLayer()(decoder_output) if\
                     cfg.Model.differential_input else decoder_output
    decoder = tf.keras.Model(inputs=inp, outputs=decoder_output, name='decoder')
    decoder.summary()
    return decoder


class VAE(tf.keras.Model):
    """Variational Autoencoder (VAE) model.
    
    This class defines a Variational Autoencoder model composed of an encoder and a decoder,
    along with loss tracking and training/testing steps.
    
    Args:
        cfg (Namespace): Configuration object containing model specifications.
    
    Attributes:
        encoder (tf.keras.Model): The encoder network.
        decoder (tf.keras.Model): The decoder network.
        loss (tf.keras.losses.Loss): The reconstruction loss function.
        kl_weights (tuple): A tuple containing (kl_weight, kl_weight_start, kl_decay_rate).
        total_loss_tracker (tf.keras.metrics.Mean): Tracker for total loss during training.
        reconstruction_loss_tracker (tf.keras.metrics.Mean): Tracker for reconstruction loss during training.
        kl_loss_tracker (tf.keras.metrics.Mean): Tracker for KL divergence loss during training.
        val_total_loss_tracker (tf.keras.metrics.Mean): Tracker for total loss during validation.
        val_reconstruction_loss_tracker (tf.keras.metrics.Mean): Tracker for reconstruction loss during validation.
        val_kl_loss_tracker (tf.keras.metrics.Mean): Tracker for KL divergence loss during validation.
    
    Methods:
        train_step(data): Perform a training step.
        test_step(data): Perform a testing/validation step.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.encoder = build_encoder(cfg)
        self.decoder = build_decoder(cfg)
        self.recon_loss = getattr(tf.keras.losses, cfg.Train.loss_type)
        self.kl_weights = (cfg.Train.kl_weight,
                           cfg.Train.kl_weight_start, 
                           cfg.Train.kl_decay_rate)  # (kl_weight, kl_weight_start, kl_decay_rate)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss")
        self.val_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="val_reconstruction_loss"
        )
        self.val_kl_loss_tracker = tf.keras.metrics.Mean(name="val_kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]

    def train_step(self, data):
        """Perform a training step.

        Args:
            data (tensor): Input data batch.

        Returns:
            Dictionary containing loss values for tracking.
        """
        kl_weight = self.kl_weights[0]
        kl_weight_start = self.kl_weights[1]
        kl_decay_rate = self.kl_weights[2]

        step = tf.cast(self.optimizer.iterations, tf.float32)
        klw = kl_weight - (kl_weight - kl_weight_start) * kl_decay_rate ** step
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    self.recon_loss(data, reconstruction), axis=1
                )
            )
            kl_loss = -klw * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(
            -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        )
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        """Perform a testing/validation step.

        Args:
            data (tensor): Input data batch.

        Returns:
            Dictionary containing loss values for tracking.
        """
        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                self.recon_loss(data, reconstruction), axis=1
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }

