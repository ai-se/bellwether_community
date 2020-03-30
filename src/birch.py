import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance
from scipy.cluster.hierarchy import dendrogram
#Clustering birch
from freediscovery.cluster import birch_hierarchy_wrapper
from freediscovery.cluster import Birch,BirchSubcluster
#Distance measure
from scipy.spatial.distance import euclidean
#Sklearn
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn import metrics
#Learners
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


import warnings

warnings.filterwarnings("ignore")

class birch(object):

    def __init__(self,threshold=0.5,branching_factor=20,n_clusters=None,outlier_threshold=0.5):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.outlier_threshold = outlier_threshold
        self.Birch_clusterer = Birch(threshold=self.threshold, branching_factor=self.branching_factor,
                                     n_clusters=self.n_clusters,compute_sample_indices=True)
    # Fitting the model with train_X
    def fit(self,data):
        self.data = data
        #self.data.drop(self.data.columns[len(self.data.columns)-1], axis=1, inplace=True)
        self.Birch_clusterer.fit(self.data)

    #Defines and builds the Cluster Feature Tree
    def get_cluster_tree(self):
        self.htree, n_clusters = birch_hierarchy_wrapper(self.Birch_clusterer)
        clusters = {}
        max_depth = 0
        for i in range(n_clusters):
            node = bcluster()
            sub_cluster = self.htree.flatten()[i]
            node.set_cluster_id(sub_cluster['cluster_id'])
            depth = sub_cluster.current_depth
            node.set_depth(depth)
            if depth > max_depth:
                max_depth = depth
            if i not in clusters.keys():
                clusters[i] = {}
            if sub_cluster.current_depth == 0:
                node.set_parent()
            else:
                node.set_parent(clusters[sub_cluster.parent['cluster_id']])
            cluster_size = sub_cluster['cluster_size']
            node.set_size(cluster_size)
            data_points = sub_cluster['document_id_accumulated']
            data_points_names = self.data.iloc[data_points].index.values.tolist()
            node.set_data_points(data_points_names)
            centroid = self.data.iloc[sub_cluster['document_id_accumulated'], :].mean(axis=0).values
            node.set_centroid(centroid)
            d1,d1_v = self.calculate_d1(centroid,data_points)
            d2 = self.calculate_d2(centroid,data_points,d1_v)
            node.add_d1(d1)
            node.add_d2(d2)
            node.calculate_threshold(self.outlier_threshold)
            clusters[i] = node
            self.cluster_tree = clusters
        return self.cluster_tree,max_depth
    
    #Calculate the d1 distance(point farthest away from centroid)
    def calculate_d1(self,centroid,data_points):
        d1 = 0
        u = centroid
        d1_v = None
        for point in data_points:
            v = point
            distance = euclidean(u,v)
            if distance>d1:
                d1 = distance
                d1_v = v
        return d1,d1_v
    
    #Calculate the d2 distance(point farthest away from d1 and its distance from centroid)
    def calculate_d2(self,centroid,data_points,d1_v):
        d2_d1 = 0
        u = d1_v
        d2_v = None
        for point in data_points:
            v = point
            distance = euclidean(u,v)
            if distance>d2_d1:
                d2_d1 = distance
                d2_v = v
        d2 = euclidean(centroid,v)
        return d2
    
    # Display's the tree
    def show_clutser_tree(self):
        self.htree.display_tree()
        
        
    # Prediction Function with height based prediction with outlier detection
    def predict(self,test_X,depth):
        predicted = []
        for test_instance in test_X.iterrows():
            test_sample = test_instance[1].values
            min_distance = float('inf')
            selected_cluster = None
            for cluster_id in self.cluster_tree:
                if self.cluster_tree[cluster_id].depth != depth:
                    continue
                u = self.cluster_tree[cluster_id].centroid
                v = np.asarray(test_sample,dtype='float64')
                distance = euclidean(u,v)
                if distance < min_distance:
                    min_distance = distance
                    selected_cluster = cluster_id
            self.cluster_tree[selected_cluster].add_test_points(test_instance[0])
            # Outlier identifier
            #if self.cluster_tree[selected_cluster].check_outlier(min_distance):
            #    self.cluster_tree[selected_cluster].add_outlier_points(test_instance[0])
            #_predicted_label = self.cluster_tree[selected_cluster].classifier.predict([test_sample])
            #self.cluster_tree[selected_cluster].add_predicted(_predicted_label)
            predicted.append(selected_cluster)
        return predicted
    
    # Model certification creator
    def certify_model(self,cluster_tree,test_y):
        for cluster_id in cluster_tree:
            if len(cluster_tree[cluster_id].test_points) == 0:
                continue
            cluster_tree[cluster_id].set_test_labels(test_y[cluster_tree[cluster_id].test_points].values)
            precision = metrics.precision_score(cluster_tree[cluster_id].test_labels, 
                                                cluster_tree[cluster_id].predicted,average='weighted')
            recall = metrics.recall_score(cluster_tree[cluster_id].test_labels, 
                                          cluster_tree[cluster_id].predicted,average='weighted')
            f1_Score = metrics.f1_score(cluster_tree[cluster_id].test_labels, 
                                        cluster_tree[cluster_id].predicted,average='weighted')
            score = {'precision': precision,'recall': recall,'f1_Score': f1_Score}
            cluster_tree[cluster_id].set_score(score)



class bcluster(object):
    
    def __init__(self):
        self.parent = None
        self.parent_id = None
        self.depth = None
        self.size = None
        self.cluster_id = None
        self.data_points = []
        self.test_points = []
        self.test_labels = []
        self.predicted = []
        self.centroid = None
        self.classifier = None
        self.outlier_model = None
        self.cluster_obj = None
        self.outlier_points = []
        self.score = []
        self.d1 = None
        self.d2 = None
        self.threshold = None
    
    def set_parent(self,parent_node=None):
        if parent_node == None:
            self.parent = None
            self.parent_id = None
        else:
            self.parent = parent_node
            self.parent_id = parent_node.cluster_id
    
    def set_depth(self,depth):
        self.depth = depth
    
    def set_size(self,size):
        self.size = size
        
    def set_cluster_id(self,cluster_id):
        self.cluster_id = cluster_id
        
    def set_data_points(self,data_points):
        self.data_points = data_points
    
    def set_test_labels(self,test_labels):
        self.test_labels = test_labels
        
    def add_test_points(self,test_point):
        self.test_points.append(test_point)
        
    def add_predicted(self,predicted):
        self.predicted.append(predicted)
    
    def set_centroid(self,centroid):
        self.centroid = centroid
        
    def set_classifier(self,classifier):
        self.classifier = classifier
        
    def set_outlier_model(self,outlier_model):
        self.outlier_model = outlier_model
        
    def set_cluster_obj(self,cluster_obj):
        self.cluster_obj = cluster_obj
        
    def add_outlier_points(self,outlier_points):
        self.outlier_points.append(outlier_points)
        
    def set_score(self,score):
        self.score = score
        
    def add_d1(self,d1):
        self.d1 = d1
        
    def add_d2(self,d2):
        self.d2 = d2
        
    def calculate_threshold(self,outlier_threshold):
        self.threshold = max(self.d1,self.d2)*outlier_threshold
        
    def check_outlier(self,distance):
        if self.threshold < distance:
            result = True
        else:
            result = False
        return result