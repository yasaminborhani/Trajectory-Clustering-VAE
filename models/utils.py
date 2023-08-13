import tensorflow as tf 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, r2_score
mpl.style.use('seaborn-v0_8')


class CustomCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for monitoring and visualization during training.
    
    Args:
        data (tf.data.Dataset): Input data for evaluating the model.
        cluster_num (int): Number of clusters for KMeans clustering.
        checkpoint_manager (tf.train.CheckpointManager): Checkpoint manager for saving weights.
        checkpoint (tf.train.Checkpoint): Checkpoint instance for restoring weights.
        cfg (YourConfigClass): Configuration class for accessing parameters.
        mode (str): 'max' for maximizing metrics, 'min' for minimizing.
    """
    def __init__(self, data, cluster_num, checkpoint_manager, checkpoint, cfg, mode='max', **kwargs):
        super(CustomCallback, self).__init__(**kwargs)
        self.data        = data
        self.cluster_num = cluster_num
        self.ts          = cfg.Model.time_shift
        self.monitor     = cfg.Train.callback_monitor
        self.manager     = checkpoint_manager
        self.score       = float('-inf') if mode == 'max' else float('inf')
        self.mode        = mode
        self.checkpoint  = checkpoint

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each training epoch.
        
        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary containing the training metrics.
        """
        # Perform KMeans clustering on z_mean
        z_mean, _, z = self.model.encoder.predict(self.data, verbose=0)
        kmeans = KMeans(n_clusters=self.cluster_num, random_state=0, n_init="auto").fit(z_mean)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(z_mean, kmeans.labels_)
        
        # Predictions and true data for r2_score calculation
        output = self.model.predict(self.data, verbose=0)
        data_pred = tf.reshape(output[-2], (output[-1].shape[0], -1))
        data_true = tf.reshape(output[-1], (output[-2].shape[0], -1)) if self.ts == 0 else\
                    tf.reshape(output[-1][:, :-self.ts, :], (output[-2].shape[0], -1))
        
        # Store metrics in logs
        logs['silhouette_score'] = silhouette_avg
        logs['r2_score'] = r2_score(data_true, data_pred)
        logs['r2_silhouette_mean'] = (silhouette_avg + r2_score(data_true, data_pred)) / 2

        # Check if metric improved and save weights
        if (self.mode == 'max' and logs[self.monitor] > self.score) or (self.mode == 'min' and logs[self.monitor] < self.score):
            print(f'\n\nResults improved from {self.score} to {logs[self.monitor]} for metric {self.monitor}, saving weights\n\n\n')
            self.manager.save()
            self.score = logs[self.monitor]

    def on_train_end(self, logs=None):
        """
        Called at the end of training.
        
        Args:
            logs (dict): Dictionary containing the training metrics.
        """
        # Restore the model with the best weights
        self.checkpoint.restore(self.manager.latest_checkpoint)
        history = self.model.history.history

        # Plot loss curves
        plt.suptitle('Losses')
        keys = [key for key in list(history.keys()) if 'loss' in key and 'val' not in key]
        for i, key in enumerate(keys):
            plt.subplot(3,1,i+1)
            plt.plot(history.history[key], color='firebrick', linewidth=3, label='Training')
            plt.plot(history.history['val_' + key], color='seagreen', linewidth=3, label='Validation')
            if i <= len(keys)-1: plt.xticks(color='w')
            plt.ylabel(key.upper())
            plt.grid('on')
            plt.legend()
        plt.subplots_adjust(hspace=0.05)
        plt.savefig('losses.pdf')
        plt.show()

        # Plot metrics curves
        plt.suptitle('Metrics')
        keys = [key for key in list(history.keys()) if 'loss' not in key]
        for i, key in enumerate(keys):
            plt.subplot(3,1,i+1)
            plt.plot(history.history[key], color='seagreen', linewidth=3, label='Validation')
            plt.xlabel('Epochs')
            plt.ylabel(key.upper())
            if i <= len(keys)-1: plt.xticks(color='w')
            plt.grid('on')
            plt.legend()
        plt.subplots_adjust(hspace=0.05)
        plt.savefig('metrics.pdf')
        plt.show()

