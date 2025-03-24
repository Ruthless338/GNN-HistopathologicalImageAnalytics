
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def visualize_features(features, domain, num_classes):
    tsne = TSNE(n_components=2)
    features_2d = tsne.fit_transform(features.cpu().numpy())
    kmeans = KMeans(n_clusters=num_classes)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=kmeans.labels_, label=domain, alpha=0.6)


