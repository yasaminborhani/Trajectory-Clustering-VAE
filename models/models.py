import tensorflow as tf
from .layers import *
from .utils import KMeansTF

def build_decoder_inputs(cfg):
    latent_inp = tf.keras.layers.Input((cfg.Model.latent), name='z_input')
    decoder_states= [] 
    multiplier = 2 if not cfg.Model.LSTM.decoder_bidirectional else 4
    for i, unit in enumerate(cfg.Model.LSTM.decoder_units):
        initial_state = tf.keras.layers.Dense(units=multiplier * unit,
                                            trainable=True if \
                                              cfg.Model.LSTM.decoder_build_init_state else False,
                                            kernel_initializer='glorot_uniform' if\
                                              cfg.Model.LSTM.decoder_build_init_state else 'zeros',
                                            use_bias=True if \
                                              cfg.Model.LSTM.decoder_build_init_state else False,
                                            activation=cfg.Model.activation,
                                            name=f'dec_initial_state_{i+1}')(latent_inp)
        states = tf.keras.layers.Lambda(lambda x: tf.split(x, multiplier, 1))(initial_state)
        decoder_states += states
    
    direct_inp = tf.keras.layers.Input((cfg.Model.temporal, cfg.Model.num_feat))
    x          = direct_inp[:, :-cfg.Model.time_shift, :] if cfg.Model.time_shift>0 else direct_inp
    if cfg.Model.decoder_input_type=='tiled':
        x_2 = tf.keras.layers.Lambda(lambda x:tf.tile(tf.expand_dims(x, 1), [1, cfg.Model.temporal - cfg.Model.time_shift, 1]))(latent_inp)
    elif cfg.Model.decoder_input_type=='upsampled':
        x_2 = tf.keras.layers.Dense((cfg.Model.temporal - cfg.Model.time_shift)* cfg.Model.latent, activation=cfg.Model.activation)(latent_inp)
        x_2 = tf.keras.layers.Reshape(((cfg.Model.temporal - cfg.Model.time_shift), cfg.Model.latent))(x_2) 
    
    dec_input = tf.keras.layers.Concatenate(axis=-1)((x_2, x)) if cfg.Model.decoder_combined_input else x_2



    decoder_inputs_model = tf.keras.Model([latent_inp, direct_inp],
                                          [dec_input, *decoder_states],
                                           name='decoder_inputs_model')
    decoder_inputs_model.summary()
    return decoder_inputs_model 
    
def build_encoder(cfg):
    """Build the encoder network based on the provided configuration.
    
    This function constructs an encoder network based on the given configuration parameters.
    
    Args:
        cfg (Namespace): Configuration object containing model specifications.
    
    Returns:
        tf.keras.Model: The constructed encoder model.
    """
    inp = tf.keras.layers.Input((cfg.Model.temporal, cfg.Model.num_feat))
    x   = inp[:, cfg.Model.time_shift:, :] if cfg.Model.time_shift>0 else inp
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
            x = AngularPositionEncoder(cfg.Model.temporal-cfg.Model.time_shift, cfg.Model.Transformer.d_model)(x)
        elif cfg.Model.Transformer.position_encoder == 'embedding' and cfg.Model.Transformer.encoder_encoding:
            x = PositionEncoder(cfg.Model.temporal-cfg.Model.time_shift, cfg.Model.Transformer.d_model)(x)
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
    feat_dim = cfg.Model.latent if not cfg.Model.decoder_combined_input\
               else cfg.Model.latent + cfg.Model.num_feat
    inp = tf.keras.layers.Input((cfg.Model.temporal-cfg.Model.time_shift,feat_dim))
    x = inp
    act = getattr(tf.nn, cfg.Model.activation)
    init_states = []
    multiplier  = 2 if not cfg.Model.LSTM.decoder_bidirectional else 4
    for i, unit in enumerate(cfg.Model.LSTM.decoder_units):
        for j in range(multiplier//2):
            init_states += [tf.keras.layers.Input((unit,), name=f'init_c{j+1}_{i+1}'),
                            tf.keras.layers.Input((unit,), name=f'init_h{j+1}_{i+1}')]

    # x = tf.keras.layers.Dense(cfg.Model.temporal * cfg.Model.num_feat, activation=act)(x)
    # x = tf.keras.layers.Reshape((cfg.Model.temporal, cfg.Model.num_feat))(x)  
    
    if cfg.Model.decoder_type == 'LSTM':
        units = cfg.Model.LSTM.decoder_units 
        return_seq = [True] * (len(units))
        for i, (unit, rt_seq) in enumerate(zip(units, return_seq)):
            x = Wrapper(tf.keras.layers.LSTM(units=unit, 
                                     activation=act,
                                     dropout=cfg.Model.LSTM.decoder_dropout_rate,
                                     unroll=cfg.Model.LSTM.unroll, 
                                     return_sequences=rt_seq), 
                                     cfg.Model.LSTM.decoder_bidirectional)(x,
                                                                           initial_state=init_states[i*multiplier:(i+1)*multiplier])


    elif cfg.Model.decoder_type == 'Transformer':
        if cfg.Model.Transformer.position_encoder == 'angular' and cfg.Model.Transformer.decoder_encoding:
            x = AngularPositionEncoder(cfg.Model.temporal-cfg.Model.time_shift, cfg.Model.Transformer.d_model)(x)
        elif cfg.Model.Transformer.position_encoder == 'embedding' and cfg.Model.Transformer.decoder_encoding:
            x = PositionEncoder(cfg.Model.temporal-cfg.Model.time_shift, cfg.Model.Transformer.d_model)(x)
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
    decoder = tf.keras.Model(inputs=[inp, *init_states], outputs=decoder_output, name='decoder')
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
        self.dec_inp = build_decoder_inputs(cfg)
        self.recon_loss = getattr(tf.keras.losses, cfg.Train.loss_type)
        self.supvis_loss = getattr(tf.keras.losses, cfg.Train.SelfSupVis.loss)
        self.apply_supervision = cfg.Train.SelfSupVis.apply_supervision
        self.kl_weights = (cfg.Train.kl_weight,
                           cfg.Train.kl_weight_start, 
                           cfg.Train.kl_decay_rate)  # (kl_weight, kl_weight_start, kl_decay_rate)
        self.sv_weights = (cfg.Train.SelfSupVis.start_weight,
                           cfg.Train.SelfSupVis.end_weight, 
                           cfg.Train.SelfSupVis.decay_rate)  # (sv_weight, sv_weight_start, sv_decay_rate)

        self.ts         = cfg.Model.time_shift
        self.return_inputs_on_call = True
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
        self.cfg = cfg
    

        if cfg.Train.SelfSupVis.apply_supervision:
            self.gmm_layers = [GMM(num_clusters, projection_dim, name=f'gmm_w_{num_clusters}_clusters') for (num_clusters, projection_dim) in zip(cfg.Train.SelfSupVis.num_clusters, cfg.Train.SelfSupVis.projection_dim)]
            self.clustering_supervision = [KMeansTF(num_clusters) for num_clusters in cfg.Train.SelfSupVis.num_clusters]


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

    def self_supervision_step(self, z_mean):
        loss = 0.0
        for i in range(len(self.cfg.Train.SelfSupVis.num_clusters)):
            y_true = tf.squeeze(tf.one_hot(tf.cast(self.clustering_supervision[i].predict(z_mean), dtype=tf.int32), depth=self.cfg.Train.SelfSupVis.num_clusters[i]))
        
            y_pred = self.gmm_layers[i](z_mean)
            
            loss =  loss + self.supvis_loss(y_true, y_pred)

        sv_weight = self.sv_weights[0]
        sv_weight_start = self.sv_weights[1]
        sv_decay_rate = self.sv_weights[2]

        step = tf.cast(self.optimizer.iterations, tf.float32)
        svw = sv_weight - (sv_weight - sv_weight_start) * sv_decay_rate ** step

        return svw * tf.reduce_mean(loss)


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
            dec_inputs     = self.dec_inp((z, data), training=True)
            reconstruction = self.decoder(dec_inputs, training=True)
            strided_data   = data[:, :-self.ts, :] if self.ts>0 else data
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    self.recon_loss(strided_data, reconstruction), axis=1
                )
            )
            kl_loss = -klw * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
            if self.apply_supervision:
                supervision_loss = self.self_supervision_step(z_mean)
                total_loss = total_loss + supervision_loss

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
    
    def call(self, x, training=False):
        z_mean, z_log_var, z = self.encoder(x, training=training)
        dec_inputs     = self.dec_inp((z, x), training=training)
        reconstruction = self.decoder(dec_inputs, training=training)
        if not self.return_inputs_on_call:
            return (z_mean, z_log_var, z, dec_inputs, reconstruction) 
        else:
            return (z_mean, z_log_var, z, dec_inputs, reconstruction, x) 
            

    def test_step(self, data):
        """Perform a testing/validation step.

        Args:
            data (tensor): Input data batch.

        Returns:
            Dictionary containing loss values for tracking.
        """
        z_mean, z_log_var, z = self.encoder(data, training=False)
        dec_inputs     = self.dec_inp((z, data), training=False)
        reconstruction = self.decoder(dec_inputs, training=False)
        strided_data   = data[:, :-self.ts, :] if self.ts>0 else data
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                self.recon_loss(strided_data, reconstruction), axis=1
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        if self.apply_supervision:
                supervision_loss = self.self_supervision_step(z_mean)
                total_loss = total_loss + supervision_loss


        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.val_total_loss_tracker.result(),
            "reconstruction_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }

