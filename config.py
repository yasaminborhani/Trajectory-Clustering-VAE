import os, re
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Preprocess setting
_C.Preprocess                           = CN() 
_C.Preprocess.normalization_method      = 'zmeanminmax' # 'standardization' / 'minmax' / 'zmeanminmax'
_C.Preprocess.split                     = False 
_C.Preprocess.split_ratio               = 0.8 
_C.Preprocess.generator                 = 'tf_pipeline'

# Training setting
_C.Train = CN()
_C.Train.epochs                         = 100
_C.Train.batch_size                     = 128
_C.Train.prefetch_buffer_size           = 256
_C.Train.shuffle_buffer_size            = 512
_C.Train.init_learning_rate             = 0.001
_C.Train.optimizer                      = 'Adam' # 'Adam' / 'SGD'/ ...
_C.Train.loss_type                      = 'mae'  # 'mae' / 'mse'
_C.Train.kl_weight                      = 0.5
_C.Train.kl_weight_start                = 0.01
_C.Train.kl_decay_rate                  = 0.99995
_C.Train.meta_data                      = "/kaggle/input/argoverse-processed-dataset/file_train.txt"

# Validation setting
_C.Valid                                = CN()
_C.Valid.batch_size                     = 128
_C.Valid.prefetch_buffer_size           = 256
_C.Valid.meta_data                      = "/kaggle/input/argoverse-processed-dataset/file_val.txt"

# Model setting 
_C.Model                                = CN()
_C.Model.latent                         = 12
_C.Model.temporal                       = 60
_C.Model.num_feat                       = 2
_C.Model.activation                     = 'tanh'   # 'tanh' / 'leaky_relu'/ 'gelu'/...
_C.Model.encoder_type                   = 'Transformer'   # 'LSTM' / 'Transformer'
_C.Model.decoder_type                   = 'LSTM'   # 'LSTM' / 'Transformer'
_C.Model.DifferenceLayer                = False

_C.Model.Transformer                    = CN()
_C.Model.Transformer.position_encoder   = 'angular'# 'angular' / 'embedding'
_C.Model.Transformer.encoder_encoding   = False
_C.Model.Transformer.decoder_encoding   = False
_C.Model.Transformer.encoder_blocks     = 1
_C.Model.Transformer.decoder_blocks     = 1
_C.Model.Transformer.num_heads          = 8
_C.Model.Transformer.d_model            = 128
_C.Model.Transformer.encoder_droprate   = 0.1
_C.Model.Transformer.decoder_droprate   = 0.1
_C.Model.Transformer.res_connection     = False

_C.Model.LSTM                           = CN()
_C.Model.LSTM.unroll                    = False
_C.Model.LSTM.encoder_units             = [128, 64]
_C.Model.LSTM.decoder_units             = [32, 64]
_C.Model.LSTM.encoder_dropout_rate      = 0.1
_C.Model.LSTM.decoder_dropout_rate      = 0.1

# Visualization setting
_C.Visualize                            = CN()
_C.Visualize.perplexity_value           = 50
_C.Visualize.n_cluster_range            = 3
