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
        history = self.model.history

        # Plot loss curves
        plt.suptitle('Losses')
        keys = [key for key in list(history.history.keys()) if 'loss' in key and 'val' not in key]
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
        keys = [key for key in list(history.history.keys()) if 'loss' not in key]
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


class SupervisionCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, num_data, cfg):
        super(SupervisionCallback, self).__init__(**kwargs)
        self.data            = data
        self.num_data        = num_data.numpy()
        self.batch_size      = cfg.Train.batch_size
        self.start_epoch     = cfg.Train.SelfSupVis.start_epoch
        self.train_frequency = cfg.Train.SelfSupVis.update_frequency
        self.input_shape     = (cfg.Preprocess.temporal, cfg.Preprocess.num_feat)

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 == self.start_epoch:
            z_mean, _, z = self.model.encoder.predict(self.data, steps = self.num_data/self.batch_size, verbose=0)
            for i in range(len(self.model.clustering_supervision)):
                self.model.clustering_supervision[i].fit(z_mean)
                c_cluster = self.model.clustering_supervision[i].cluster_centers_
                init_weight = self.model.gmm_layers[i].get_weights()
                new_weights = [tf.transpose(c_cluster), *init_weight[1:]]
                self.model.gmm_layers[i].set_weights(new_weights)
        
            
        if epoch + 1 > self.start_epoch and epoch % self.train_frequency == 0:
            z_mean, _, z = self.model.encoder.predict(self.data, steps = self.num_data/self.batch_size, verbose=0)
            for i in range(len(self.model.clustering_supervision)):
                self.model.clustering_supervision[i].fit(z_mean)




class KMeansTF:
    def __init__(self, n_clusters, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, X):
        num_samples, num_features = X.shape

        if self.random_state is not None:
            tf.random.set_seed(self.random_state)

        if self.cluster_centers_ is not None:
            initial_indices = tf.random.shuffle(tf.range(num_samples))[:self.n_clusters]
            centroids = tf.Variable(self.cluster_centers_)
        else:
            initial_indices = tf.random.shuffle(tf.range(num_samples))[:self.n_clusters]
            centroids = tf.Variable(tf.gather(X, initial_indices))

        def compute_labels(centroids):
            distances = tf.reduce_sum(tf.square(X[:, tf.newaxis] - centroids), axis=2)
            labels = tf.argmin(distances, axis=1)
            return labels

        def compute_centroids(labels):
            new_centroids = tf.stack([tf.reduce_mean(tf.boolean_mask(X, tf.equal(labels, i)), axis=0) for i in range(self.n_clusters)])
            return new_centroids




        iter_count = tf.constant(0)
        def loop_condition(iter_count, centroids):
            return tf.less(iter_count, self.max_iters)

        def loop_body(iter_count, centroids):
            labels = compute_labels(centroids)
            new_centroids = compute_centroids(labels)
            return iter_count + 1, new_centroids

        _, final_centroids = tf.while_loop(
            loop_condition,
            loop_body,
            [iter_count, centroids],
            parallel_iterations=1
        )

        final_labels = compute_labels(final_centroids)
        self.labels_ = final_labels.numpy()
        self.cluster_centers_ = final_centroids.numpy()

    def predict(self, X_new):
        distances = tf.reduce_sum(tf.square(X_new[:, tf.newaxis] - self.cluster_centers_), axis=2)
        predicted_labels = tf.argmin(distances, axis=1).numpy()
        return predicted_labels