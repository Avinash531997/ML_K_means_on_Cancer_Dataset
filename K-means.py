#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd 

style.use('ggplot')

class KMeans: # Defining KMeans Class 
    def __init__(self, k =2, tol = 0.001, max_iter = 400): # constructor
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    # Function for Assigning points to appropriate Clusters
    def fit(self, data):        

        self.Centroid = {}

        for i in range(self.k): 
            self.Centroid[i] = data[i]
            
        for i in range(self.max_iter):
            self.Class = {}
            for i in range(self.k):
                self.Class[i] = []
            
            #Calculate  the distance between the data points and Centroid of the cluster, and 
            #Choose the closest Centroid
            for features in data: 
                distances = [np.linalg.norm(features - self.Centroid[CENTROID]) for CENTROID in self.Centroid]
                classification = distances.index(min(distances))
                self.Class[classification].append(features)

            prev = dict(self.Centroid) 
            
            #Calculate the weighted average of the cluster Data Points 
            #calculating the Centroid again
            for classification in self.Class: 
                self.Centroid[classification] = np.average(self.Class[classification], axis = 0) 

            isOptimal = True
            
            #Update the  values of Centroid 
            #check  whether the  Centroid are changed 
            for CENTROID in self.Centroid: 
                original_CENTROID = prev[CENTROID]
                curr = self.Centroid[CENTROID]
                if np.sum((curr - original_CENTROID)/original_CENTROID * 100.0) > self.tol: 
                    isOptimal = False
            
            if isOptimal:
                break
    
if __name__ == "__main__":
    
    #Receiving the Data Set as input
    data = pd.read_csv('cancer.csv')
    # Removes first two attributes
    df = data.iloc[:,2:32] 
    
    # Removing all labels and taking values only
    x = df.values  
    k=2 # Number of Clusters Considered
    km = KMeans(k) 
    km.fit(x)
    CENTER_Color = ['red','blue'] # It is used for providing color to centroid
    CLUSTER_POINT_COLOR = ["orange", "green"] # It is used for providing color to data points of each cluster
    
    CLUSTER_1 = 0
    CLUSTER_2 = 0
    
    #Assign color to each point with it's respective Centroid
    for classification in km.Class:
        color_assign = CLUSTER_POINT_COLOR[classification]
        for features in km.Class[classification]:  # Loop and Verify to which cluster the Data point belongs
            if color_assign == 'orange':  
                CLUSTER_1 =CLUSTER_1 + 1
            else:     
                CLUSTER_2 =CLUSTER_2 + 1
            plt.scatter(features[0], features[1], color = color_assign,s = 25) 
    
    # Assign a color to each Centroid in cluster
    for CENTROID in km.Centroid:
        plt.scatter(km.Centroid[CENTROID][0], km.Centroid[CENTROID][1],c=CENTER_Color[CENTROID], s = 300, marker = "*") 
   
    #Plotting
    plt.title('Clusters Formed with  K-Means for  K=2')
    plt.xlabel('Radius_Mean')
    plt.ylabel('Texture_Mean')
    plt.savefig('Kmeans.png')
    plt.show()   
            
    print("No. of Data points in Cluster 1 < Orange cluster> : " + str(CLUSTER_1))
    print("No. of Data points in Cluster 2 < Green  cluster>: " + str(CLUSTER_2))


# In[ ]:




