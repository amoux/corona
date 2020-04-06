import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

epochs = 300
tol = 1e-12
seed = 0
n_centroids = 10
maxclusters = 13
embedd_file = '../data/cluster_data/embedd_biorxiv_medrxiv.npz'

if __name__ == "__main__":
    distortions = []
    embedding = np.load(embedd_file)['embedding']
    for cluster in tqdm(range(1, maxclusters), desc="clusters"):
        km = KMeans(cluster, init='random',
                    n_init=n_centroids, tol=tol,
                    max_iter=epochs, random_state=seed)
        km.fit(embedding)
        distortions.append(km.inertia_)

    plt.plot(range(1, maxclusters), distortions, marker='o')
    plt.xlabel('clusters')
    plt.ylabel('distortions')
    plt.savefig(f'{maxclusters}_cluster_distortions.png')
