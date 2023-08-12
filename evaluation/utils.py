import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from ..datasets.agroverse.utils import unnormalized_data

def create_bar_chart(data_dict, title):
    categories = list(data_dict.keys())
    values = list(data_dict.values())

    # Choose a colormap from Matplotlib (e.g., 'viridis', 'plasma', 'tab20', 'Set2', etc.)
    colormap = 'Set2'

    plt.bar(categories, values, color=plt.cm.get_cmap(colormap)(range(len(categories))))
    plt.xlabel('Categories')
    plt.ylabel('Value')
    plt.title(title)
    plt.xticks(rotation=45)  # Rotate the category labels for better visibility
    plt.grid(axis='y')  # Add grid lines on the y-axis
    plt.show()

def create_grouped_bar_chart(data_dict1, data_dict2, title):
    categories = list(data_dict1.keys())
    values1 = list(data_dict1.values())
    values2 = list(data_dict2.values())

    # Set the width of the bars
    bar_width = 0.4

    # Calculate the positions of the bars on the x-axis
    bar_positions1 = np.arange(len(categories))
    bar_positions2 = bar_positions1 + bar_width

    plt.bar(bar_positions1, values1, width=bar_width, label='Train')
    plt.bar(bar_positions2, values2, width=bar_width, label='Validation')

    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title(title)
    plt.xticks(bar_positions1 + bar_width / 2, categories, rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.show()

def plot_clusters(kmeans, z_mean, cluster_centers):
    # Plotting the cluster centers (centroids)
#     cluster_centers = kmeans.cluster_centers_

    # Get cluster assignments for each data point
    cluster_assignments = kmeans.labels_

    # Create a scatter plot of the data points, colored by cluster assignments
    for cluster_id in np.unique(cluster_assignments):
        plt.scatter(z_mean[cluster_assignments == cluster_id][:, 0], z_mean[cluster_assignments == cluster_id][:, 1], label=f'Cluster {cluster_id}')

    # Scatter plot the cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red', marker='X', label='Cluster Centers')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.grid(True)
    plt.show()


def clustering(model, data_gen, train_flag, num_train_samples, cfg):
    if train_flag:
        z_mean, _, _ = model.encoder.predict(data_gen, steps = num_train_samples.numpy() // cfg.Train.batch_size)
    else:
        z_mean, _, _ = model.encoder.predict(data_gen)

    # Adjust the perplexity as needed
    tsne = TSNE(n_components=2, perplexity=cfg.Visualize.perplexity_value, random_state=42)

    silhouette_met = {}
    davies_bouldin_met = {}


    for i in cfg.Visualize.n_cluster_range:
        kmeans          = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(z_mean)
        cluster_centers = kmeans.cluster_centers_
        n_clusters      = cluster_centers.shape[0]

        z_tsne          = tsne.fit_transform(np.concatenate((z_mean,
                                                             cluster_centers),
                                                             axis=0))


        silhouette_avg = silhouette_score(z_mean, kmeans.labels_)
        silhouette_met[str(i)+' clusters'] = silhouette_avg

        davies_bouldin = davies_bouldin_score(z_mean, kmeans.labels_)
        davies_bouldin_met[str(i)+' clusters'] = davies_bouldin

        plot_clusters(kmeans, z_tsne[:-n_clusters, :], z_tsne[-n_clusters:, :])
    return silhouette_met, davies_bouldin_met
    
    
def plot_trajectory_samples(model, silhouette_score, data_gen, train_flag, num_train_samples, cfg):
    if train_flag:
        z_mean, _, _ = model.encoder.predict(data_gen, steps = num_train_samples.numpy() // cfg.Train.batch_size)
    else:
        z_mean, _, _ = model.encoder.predict(data_gen)
        
    scores= list(silhouette_score.values())
    best_score_idx = scores.index(max(scores))
    n_cluster = int(list(silhouette_score.keys())[best_score_idx].split()[0])

    kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init="auto").fit(z_mean)

    data_batch         = next(iter(data_gen))
    z_mean_batch, _, _ = model.encoder.predict(data_batch)
    labels             = kmeans.predict(z_mean_batch)

    for i in range(n_cluster):
        data = data_batch.numpy()[labels==i,:,:]
        data = unnormalized_data(data, cfg.Preprocess.normalization_method, normalization_params)

        shape_data = data.shape
        if shape_data[0] != 0:
            for j in range(min(shape_data[0], 6)):
                dx = [data[j,m+1,0] - data[j, m, 0] for m in range(len(data[j,:,0]) - 1)]
                dy = [data[j,m+1,1] - data[j, m, 1] for m in range(len(data[j,:,1]) - 1)]

                num_arrows = 8
                arrow_interval = max(len(data[j,:,0]) // num_arrows, 1)

                # Create quiver coordinates and direction vectors for arrows
                quiver_x = data[j, :-1:arrow_interval, 0]
                quiver_y = data[j, :-1:arrow_interval, 1]
                quiver_dx = dx[::arrow_interval]
                quiver_dy = dy[::arrow_interval]

                # Add arrows along the trajectory using plt.quiver()
                arrow_scale = 10  # Adjust arrow scale for arrow length

                plt.suptitle('label '+ str(i))
                plt.subplot(2,3,j+1)
                plt.plot(data[j,:,0], data[j,:,1], label='Trajectory')
                plt.scatter(data[j,0,0], data[j,0,1], color='green', marker='o', s=100, label='Start')
                plt.scatter(data[j,-1,0], data[j,-1,1], color='red', marker='*', s=150, label='End')
                plt.quiver(quiver_x, quiver_y, quiver_dx, quiver_dy, angles='xy', scale_units='xy', color='red', width=0.025)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True, ls='--')
                plt.subplots_adjust(hspace=0.3, wspace=0.7)
            plt.show()


def accumulated_trajectorie(model, n_clusters, data_gen, train_flag, num_train_samples, normalization_params, cfg):
    if train_flag:
        z_mean, _, _ = model.encoder.predict(data_gen, steps = num_train_samples.numpy() // cfg.Train.batch_size)
    else:
        z_mean, _, _ = model.encoder.predict(data_gen)
   
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(z_mean)
    
    data_batch         = next(iter(data_gen))
    z_mean_batch, _, _ = model.encoder.predict(data_batch)
    labels             = kmeans.predict(z_mean_batch)

    for i in range(n_clusters):
        data = data_batch.numpy()[labels==i,:,:]
        data = unnormalized_data(data, cfg.Preprocess.normalization_method, normalization_params)

        shape_data = data.shape
        if shape_data[0] != 0:

            for j in range(50):
                try:
                    plt.suptitle('label '+ str(i)) 
                    plt.plot(data[j,:,0], data[j,:,1], label='Trajectory', color='green')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.grid(True, ls='--')

                except:
                    continue

            plt.show()


