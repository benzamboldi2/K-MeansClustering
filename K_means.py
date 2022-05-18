# Import libraries
import pandas as pd
import math
from sklearn.manifold import TSNE
import seaborn as sb
import matplotlib.pyplot as plt

# load the data
train = pd.read_csv('K_means_train.csv')
validate = pd.read_csv('K_means_valid.csv')
test = pd.read_csv('K_means_test.csv')

# Function to calculate euclidean distance between two vectors
def euclidean(point1, point2):
    sqDistance = 0
    for point in range(len(point1)):
        diff = point2[point] - point1[point]
        sqDistance = sqDistance + (diff * diff)
    return math.sqrt(sqDistance)

class K_means:

    # Initilize data, K, centroids, and clusters
    def __init__(self, data, K):
        self.data = data
        self.centroids = [[4.9, 2.5, 4.5, 1.7], [5.6, 2.5, 3.9, 1.1], [6.3, 2.9, 5.6, 1.8]]
        self.clusters = [[] for _ in range(K)]
        self.K = K
        for index, row in self.data.iterrows():
            dists = []
            for cluster in range(len(self.centroids)):
                row_i = [row['SepalLengthCm'], row['SepalWidthCm'], row['PetalLengthCm'], row['PetalWidthCm']]
                dists.append(euclidean(row_i, self.centroids[cluster]))
            self.clusters[dists.index(min(dists))].append(index)
    
    # Method to update the centroids of clusters
    def recompute_center(self):
        dim = self.data.shape[1]
        for i in range(len(self.clusters)):
            center = []
            for j in range(dim):
                sub_sum = 0
                for point in self.clusters[i]:
                    sub_sum += self.data.loc[point][j]
                mean = sub_sum / len(self.clusters[i])
                center.append(mean)
            self.centroids[i] = center

    # Method to check if the clusters changed from one iteration to the next
    def check_difference(self, cluster1, cluster2):
        for i in range(len(cluster1)):
            if (len(cluster1[i]) != len(cluster2[i])):
                return False
            else:
                for j in range(len(cluster1[i])):
                    if (cluster1[i][j] != cluster2[i][j]):
                        return False
        # If all points are the same, return true
        return True

    # Recursive method to assign the training clusters
    def assign_clusters(self):
        preClusters = self.clusters
        self.recompute_center()
        clustersTemp = [[] for _ in range(self.K)]
        for index, row in self.data.iterrows():
            dists = []
            for center in self.centroids:
                row_i = [row['SepalLengthCm'], row['SepalWidthCm'], row['PetalLengthCm'], row['PetalWidthCm']]
                dists.append(euclidean(row_i, center))
            clustersTemp[dists.index(min(dists))].append(index)
        
        if (self.check_difference(preClusters, clustersTemp) == True):
            # Base case: the clusters did not change
            self.clusters = clustersTemp
            return 0
        else:
            self.clusters = clustersTemp
            # Recursively assign points to clusters
            self.assign_clusters()
            return self.centroids 

# Main method
if __name__ == '__main__':
    trainTemp = train
    train = train.drop(['Id', 'Labels'], axis=1)
    
    # Create instance and get clusters
    K_means_train = K_means(train, 3)
    clusters = K_means_train.assign_clusters()

    testData = test.drop(['Id', 'labels'], axis=1)

    results = []
    # For each point in test, assign it to nearest cluster
    for index, row in testData.iterrows():
        dists = []
        for center in clusters:
            row_i = [row['SepalLengthCm'], row['SepalWidthCm'], row['PetalLengthCm'], row['PetalWidthCm']]
            dists.append(euclidean(row_i, center))
        results.append('cluster_' + str(dists.index(min(dists)) + 1))
    
    # Final results output
    test = test.drop(['labels'], axis=1)
    test = pd.concat((test, pd.Series(results, name='labels')), axis=1)
    print(test)

    train_embedded = TSNE(n_components=2, verbose=1, random_state=123).fit_transform(train)
    print(train_embedded.shape)

    # Get data for TSNE plot
    df = pd.DataFrame()
    labels = []
    for index, row in train.iterrows():
        dists = []
        for center in clusters:
            row_i = [row['SepalLengthCm'], row['SepalWidthCm'], row['PetalLengthCm'], row['PetalWidthCm']]
            dists.append(euclidean(row_i, center))
        labels.append('cluster_' + str(dists.index(min(dists)) + 1))    
    df['SepalLengthCm'] = trainTemp['SepalLengthCm']
    df['SepalWidthCm'] = trainTemp['SepalWidthCm']

    sb.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue=labels,
                data=df).set(title='Iris data T-SNE projection')
    plt.show()