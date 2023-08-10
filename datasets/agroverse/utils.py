import tensorflow as tf 
import numpy as np 




def parse_text(data, cfg):
    """
    Parse text data into a processed numerical array.
    
    Args:
        data (tf.Tensor): Text data tensor.
    
    Returns:
        tf.Tensor: Processed numerical array.
    """
    processed_data = tf.strings.split(data)[1:]
    processed_data = tf.strings.to_number(processed_data, tf.float32)
    processed_data = tf.reshape(processed_data, (cfg.Model.temporal, cfg.Model.num_feat))
    
    return processed_data

def matrix_min(matrix1, matrix2):
    """
    Calculate element-wise minimum of two matrices.
    
    Args:
        matrix1 (tf.Tensor): First input matrix.
        matrix2 (tf.Tensor): Second input matrix.
    
    Returns:
        tf.Tensor: Matrix containing element-wise minimum values.
    """
    return tf.reduce_min(tf.minimum(matrix1, matrix2), axis=0)

def matrix_max(matrix1, matrix2):
    """
    Calculate element-wise maximum of two matrices.
    
    Args:
        matrix1 (tf.Tensor): First input matrix.
        matrix2 (tf.Tensor): Second input matrix.
    
    Returns:
        tf.Tensor: Matrix containing element-wise maximum values.
    """
    return tf.reduce_max(tf.maximum(matrix1, matrix2), axis=0)

def reduce_fn(state, data):
    """
    Reduce function for summing data along axis 0.
    
    Args:
        state (tf.Tensor): Accumulated state tensor.
        data (tf.Tensor): Input data tensor.
    
    Returns:
        tf.Tensor: Accumulated state tensor after reduction.
    """
    return state + tf.reduce_sum(data, axis=0)

def reduce_square_sum(state, data):
    """
    Reduce function for summing the square of data along axis 0.
    
    Args:
        state (tf.Tensor): Accumulated state tensor.
        data (tf.Tensor): Input data tensor.
    
    Returns:
        tf.Tensor: Accumulated state tensor after reduction.
    """
    return state + tf.reduce_sum(tf.square(data), axis=0)

def count_samples(acc, sample):
    """
    Count the number of samples.
    
    Args:
        acc (tf.Tensor): Accumulated count tensor.
        sample (tf.Tensor): Input sample tensor.
    
    Returns:
        tf.Tensor: Accumulated count tensor after incrementing.
    """
    return acc + 1

def param_extractor(dataset, cfg):
    """
    Extract normalization parameters from the dataset.
    
    Args:
        dataset (tf.data.Dataset): Input dataset containing samples.
    
    Returns:
        tuple: Tuple containing min, max, mean, and std normalization parameters.
    """
    max_possible_value = tf.constant(float('inf'), shape=(1, cfg.Model.num_feat), dtype=tf.float32)
    min_feature = dataset.reduce(max_possible_value, lambda x, y: matrix_min(x, y))
    
    min_possible_value = tf.constant(float('-inf'), shape=(1, cfg.Model.num_feat), dtype=tf.float32)
    max_feature = dataset.reduce(min_possible_value, lambda x, y: matrix_max(x, y))
    
    initial_sum = tf.zeros(shape=(1, cfg.Model.num_feat), dtype=tf.float32)
    total_sum = dataset.reduce(initial_sum, reduce_fn)
    num_samples = dataset.reduce(tf.constant(0), lambda acc, _: count_samples(acc, _))
    mean_feature = total_sum / tf.cast(num_samples * cfg.Model.temporal, tf.float32)
    
    initial_square_sum = tf.zeros(shape=(1, cfg.Model.num_feat), dtype=tf.float32)
    total_square_sum = dataset.reduce(initial_square_sum, reduce_square_sum)
    std_feature = tf.math.sqrt(total_square_sum / tf.cast(num_samples * cfg.Model.temporal, tf.float32) - (mean_feature)**2)
    
    return min_feature[tf.newaxis, ...], max_feature[tf.newaxis, ...], mean_feature[tf.newaxis, ...], std_feature[tf.newaxis, ...]

def normalize(x, normalization_method, normalization_params):
    """
    Normalize data using specified normalization method and parameters.
    
    Args:
        x (tf.Tensor): Input data tensor to be normalized.
        normalization_method (str): Normalization method ('minmax', 'standardization', 'zmeanminmax').
        normalization_params (tuple): Tuple containing normalization parameters.
    
    Returns:
        tf.Tensor: Normalized data tensor.
    """
    min = normalization_params[0]
    max = normalization_params[1]
    mean = normalization_params[2]
    std = normalization_params[3]
    
    if normalization_method == 'minmax':
        return (x - min) / (max - min)
    elif normalization_method == 'standardization':
        return (x - mean) / (std)
    elif normalization_method == 'zmeanminmax':
        return (x - mean) / (max - min)
    else:
        return x

def unnormalized_data(data, normalization_method, normalization_params):
    """
    Restore unnormalized data using specified method and parameters.
    
    Args:
        data (tf.Tensor): Normalized data tensor to be unnormalized.
        normalization_method (str): Normalization method ('minmax', 'standardization', 'zmeanminmax').
        normalization_params (tuple): Tuple containing normalization parameters.
    
    Returns:
        tf.Tensor: Unnormalized data tensor.
    """
    min_feature = normalization_params[0]
    max_feature = normalization_params[1]
    mean_feature = normalization_params[2]
    std_feature = normalization_params[3]
    
    if normalization_method == 'standardization':
        return data * std_feature + mean_feature
    elif normalization_method == 'minmax':
        return data * (max_feature - min_feature) + min_feature
    elif normalization_method == 'zmeanminmax':
        return data * (max_feature - min_feature) + mean_feature
    else:
        return data



class DataGenerator(tf.keras.utils.Sequence):
    """
    This class is designed to efficiently generate batches of data for training or evaluation in Keras. It is specifically
    tailored for processing sequential data with multiple features, commonly used in machine learning tasks such as time
    series prediction or sequence-to-sequence models. The class can be used to process and normalize data from a given list
    of IDs, where each ID corresponds to a directory containing parquet files with sequential data.

    Parameters:
        list_IDs (list): A list of IDs corresponding to directories containing parquet files with sequential data.
        batch_size (int, optional): The number of samples in each batch. Default is 128.
        feature_num (int, optional): The number of features in each data point. Default is 2.
        time_step (int, optional): The number of time steps in each data sequence. Default is 60.
        normalization_param (tuple, optional): A tuple containing the normalization parameters (mean, std, max, min) for
                                               each feature. If any normalization parameter is set to None, it will be
                                               computed based on the data. Default is (None, None, None, None).
        normalization_method (str, optional): The normalization method to use. Options are 'minmax' for min-max scaling
                                              or 'standardization' for z-score normalization. If set to None, no
                                              normalization is applied. Default is None.
        shuffle (bool, optional): If True, the data is shuffled at the end of each epoch. Default is True.
    """

    def __init__(self, list_IDs,
                 batch_size = 128,
                 feature_num = 2,
                 time_step = 60,
                 normalization_param = (None, None, None, None), #mean, std, max, min
                 normalization_method = None, # minmax or standardization
                 shuffle = True):
        'Initialization'
        # self.dim = dim
        self.batch_size  = batch_size
        self.feature_num = feature_num
        self.time_step   = time_step
        self.list_IDs    = list_IDs
        self.shuffle     = shuffle
        self.mean        = normalization_param[0] if normalization_param[0] is not None else np.zeros((feature_num, ), dtype=np.float32)
        self.std         = normalization_param[1] if normalization_param[1] is not None else np.ones((feature_num, ), dtype=np.float32)
        self.max         = normalization_param[2] if normalization_param[2] is not None else np.zeros((feature_num, ), dtype=np.float32) + np.inf
        self.min         = normalization_param[3] if normalization_param[3] is not None else np.zeros((feature_num, ), dtype=np.float32) - np.inf
        self.normalization_method = normalization_method
        self.on_epoch_end()

    def normalize(self, x):
        if self.normalization_method == 'minmax':
            return (x - self.min) / (self.max - self.min)

        elif self.normalization_method == 'standardization':
            return (x - self.mean) / (self.std)
        elif self.normalization_method == 'zmeanminmax':
            return (x - self.mean) / (self.max - self.min)
        else:
            return x

    @classmethod
    def update_normalization_params(cls,
                                    list_IDs,
                                    batch_size = 128,
                                    feature_num = 2,
                                    time_step = 60,
                                    normalization_param = (None, None, None, None), #mean, std, max, min
                                    normalization_method = None, # minmax or standardization
                                    ):

        data_iter      = iter(cls(list_IDs,
                                  batch_size = batch_size,
                                  feature_num = feature_num,
                                  time_step = time_step))

        min_feature    = np.zeros((feature_num, ), dtype=np.float32) + np.inf
        max_feature    = np.zeros((feature_num, ), dtype=np.float32) - np.inf

        sum_data       = np.zeros((feature_num, ), dtype=np.float32)
        sum2_data      = np.zeros((feature_num, ), dtype=np.float32)



        for _ in tqdm.tqdm(range(len(list_IDs)//batch_size)):
            data, _ = next(data_iter)
            min_feature = np.minimum(data.reshape((-1, feature_num)).min(axis=0), min_feature)
            max_feature = np.maximum(data.reshape((-1, feature_num)).max(axis=0), min_feature)
            sum_data    = sum_data + np.sum(data.reshape((-1, feature_num)),axis=0)
            sum2_data   = sum2_data + np.sum((data**2).reshape((-1, feature_num)),axis=0)

        mean_feature = sum_data / (len(list_IDs) * time_step)
        std_feature  = np.sqrt((sum2_data / (len(list_IDs)*time_step)) - mean_feature**2)
        n_p          = (mean_feature, std_feature, max_feature, min_feature)

        return cls(list_IDs,
                    batch_size = batch_size,
                    feature_num = feature_num,
                    time_step = time_step,
                    normalization_param = n_p, #mean, std, max, min
                    normalization_method = normalization_method) , n_p


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.time_step, self.feature_num))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            val_dir           = ID
            _, parquet_file   = sorted(glob.glob(os.path.join(val_dir, '*')))
            df_from_parq      = pd.read_parquet(parquet_file)

            # Store sample
            X[i,] = np.squeeze(self.process_data(df_from_parq, self.feature_num))


        return self.normalize(X), self.normalize(X)

    @staticmethod
    def process_data(df_from_parq, num_feature):

        focal_agent          = df_from_parq.loc[df_from_parq['track_id']==df_from_parq['focal_track_id']]
        av_agent             = df_from_parq.loc[df_from_parq['track_id']=='AV']
        if num_feature == 2:
            focal_agent_features = focal_agent[['position_x', 'position_y', 'heading']].values[50:,:]
            av_agent_features    = av_agent[['position_x', 'position_y', 'heading']].values[50:,:]
            
            focal_agent_xy  = focal_agent_features[:,:2]
            av_agent_xy     = av_agent_features[:,:2]
            av_agent_theta  = av_agent_features[:,2]
            rotation_matrix = np.array([[np.cos(av_agent_theta), np.sin(av_agent_theta)],
                                        [-np.sin(av_agent_theta), np.cos(av_agent_theta)]])

            rotation_matrix = np.array(tf.transpose(rotation_matrix, perm=[2,0,1]))

            focal_new_features = tf.einsum('mab,mbc->mac',rotation_matrix,(focal_agent_xy - av_agent_xy)[...,np.newaxis])
            focal_new_features = np.array(focal_new_features)
    
            return focal_new_features
            
            
        elif num_feature == 4:
            focal_agent_features = focal_agent[['position_x', 'position_y', 'heading', 'velocity_x', 'velocity_y']].values[50:,:]
            av_agent_features    = av_agent[['position_x', 'position_y', 'heading', 'velocity_x', 'velocity_y']].values[50:,:]
            
            focal_agent_xy  = focal_agent_features[:,:2]
            av_agent_xy     = av_agent_features[:,:2]
            focal_agent_v   = focal_agent_features[:,-2:]
            av_agent_v      = av_agent_features[:,-2:]
            av_agent_theta  = av_agent_features[:,2]
            rotation_matrix = np.array([[np.cos(av_agent_theta), np.sin(av_agent_theta)],
                                        [-np.sin(av_agent_theta), np.cos(av_agent_theta)]])

            rotation_matrix = np.array(tf.transpose(rotation_matrix, perm=[2,0,1]))

            focal_new_features_pos = tf.einsum('mab,mbc->mac',rotation_matrix,(focal_agent_xy - av_agent_xy)[...,np.newaxis])
            focal_new_features_pos = np.array(focal_new_features_pos)
            
            focal_new_features_vel = tf.einsum('mab,mbc->mac',rotation_matrix,(focal_agent_v - av_agent_v)[...,np.newaxis])
            focal_new_features_vel = np.array(focal_new_features_vel)
            
            return np.concatenate([focal_new_features_pos, focal_new_features_vel], axis=1)
            
            
       
    
    @staticmethod
    def unnormalized_data(data, normalization_method, normalization_params):
        mean_feature = normalization_params[0]
        std_feature  = normalization_params[1]
        max_feature  = normalization_params[2]
        min_feature  = normalization_params[3]
        
        if normalization_method == 'standardization':
            return data * std_feature + mean_feature
        elif normalization_method == 'minmax':
            return data * (max_feature - min_feature) + min_feature
        elif normalization_method == 'zmeanminmax':
            return data * (max_feature - min_feature) + mean_feature
        else:
            return data