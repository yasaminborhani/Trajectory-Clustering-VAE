import tensorflow as tf 
import numpy as np 
import pandas as pd 
from .utils import *
import glob
import os

def sequential_generator(cfg):
    """
    Generate sequential data using directories.

    Args:
        cfg (Config): Configuration object containing various parameters.

    Returns:
        DataGenerator: Training data generator.
        DataGenerator: Validation data generator.
        int: Number of training directories.
        tuple: Normalization parameters for data.
    """
    dirs = sorted(glob.glob(os.path.join(cfg.Train.meta_data, '*')))
    
    if cfg.Preprocess.split:
        spl = int(len(dirs) * cfg.Preprocess.split_ratio)
        train_dirs = dirs[:spl]
        val_dirs = dirs[spl:]
    else: 
        train_dirs = dirs 
        val_dirs = sorted(glob.glob(os.path.join(cfg.Valid.meta_data, '*')))
    
    data_generator_train, normalization_param = DataGenerator.update_normalization_params(
        list_IDs=train_dirs,
        batch_size=cfg.Train.batch_size,
        feature_num=cfg.Model.num_feat,
        time_step=cfg.Model.temporal,
        normalization_param=(None, None, None, None),
        normalization_method=cfg.Preprocess.normalization_method,
    )

    data_generator_val = DataGenerator(
        list_IDs=val_dirs,
        batch_size=batch_size,
        feature_num=feature_num,
        time_step=time_step,
        normalization_param=normalization_param,
        normalization_method=normalization,
    )
    
    return data_generator_train, data_generator_val, len(train_dirs), normalization_param

def tf_pipeline_gen(cfg):
    """
    Create a TensorFlow pipeline for data preprocessing.

    Args:
        cfg (Config): Configuration object containing various parameters.

    Returns:
        tf.data.Dataset: Training dataset.
        tf.data.Dataset: Validation dataset.
        int: Number of training samples.
        tuple: Normalization parameters for data.
    """
    train_dataset = tf.data.TextLineDataset([cfg.Train.meta_data])
    train_dataset = train_dataset.map(lambda x: parse_text(x, cfg=cfg), num_parallel_calls=tf.data.AUTOTUNE)

    num_samples = train_dataset.reduce(tf.constant(0), lambda acc, _: count_samples(acc, _))
    normalization_params = param_extractor(train_dataset, cfg)

    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(cfg.Train.shuffle_buffer_size)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(cfg.Train.batch_size)
    train_dataset = train_dataset.map(lambda x: normalize(x, normalization_method=cfg.Preprocess.normalization_method, normalization_params=normalization_params), num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.prefetch(cfg.Train.prefetch_buffer_size)

    # validation
    val_dataset = tf.data.TextLineDataset([cfg.Valid.meta_data])
    val_dataset = val_dataset.map(parse_text, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.batch(cfg.Valid.batch_size)
    val_dataset = val_dataset.map(lambda x: normalize(x, normalization_method=cfg.Preprocess.normalization_method, normalization_params=normalization_params), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(cfg.Valid.prefetch_buffer_size)

    return train_dataset, val_dataset, num_train_samples, normalization_params

def make_gen(cfg):
    """
    Create a data generator based on the specified method.

    Args:
        cfg (Config): Configuration object containing various parameters.

    Returns:
        DataGenerator or tf.data.Dataset: Data generator or dataset.
    """
    if cfg.Preprocess.generator == 'tf_pipeline':
        return tf_pipeline_gen(cfg) 
    else:
        return sequential_generator(cfg)
