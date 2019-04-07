import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from graphviz import Digraph
from sklearn.metrics.pairwise import pairwise_distances
import scipy.spatial.distance
from scipy.cluster.hierarchy import dendrogram
from freediscovery.cluster import birch_hierarchy_wrapper
from freediscovery.cluster import Birch

class birch(object):

    def __init__(self,threshold=0.5,branching_factor=20,n_clusters=None):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.Birch_clusterer = Birch(threshold=self.threshold, branching_factor=self.branching_factor,n_clusters=self.n_clusters)
    
    def fit(self,data):
        self.data = data
        self.data.drop(self.data.columns[len(self.data.columns)-1], axis=1, inplace=True)
        self.Birch_clusterer.fit(self.data)

    def get_cluster_labels(self):
        htree, n_clusters = birch_hierarchy_wrapper(self.Birch_clusterer)
        clusters = {}
        for i in range(n_clusters):
            sub_cluster = htree.flatten()[i]
            if i not in clusters.keys():
                clusters[i] = {}
            if sub_cluster.current_depth == 0:
                clusters[i]['parent'] = None
            else:
                clusters[i]['parent'] = sub_cluster.parent['cluster_id']
            clusters[i]['size'] = sub_cluster['cluster_size']
            clusters[i]['data_points'] = sub_cluster['document_id_accumulated']
        return clusters
