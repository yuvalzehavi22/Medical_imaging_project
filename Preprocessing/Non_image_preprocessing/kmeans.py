import zipfile
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import os
import shutil
import argparse
import seaborn as sns
from utils import convert_categorical_to_numeric, sort_features_by_image_ids, preprocess_cat_numeric
from Clustering_evaluation import type_heatmap, cramers_v_matrix, cramers_v_patients
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
import pickle



def parse_args():
    parser = argparse.ArgumentParser(description='kmeans')
    parser.add_argument('--K', type=int, default=3, help="Number of clusters (graphs)")
    parser.add_argument('--K_images', type=int, default=3, help="Number of clusters of image data ")
    parser.add_argument('--thres', type=str, default='0.8,0.8,0.8,0.8', help="Threshold value for distance")
    parser.add_argument('--rnd', type=int, default=5, help="Random State for kmeans model")
    parser.add_argument('--dataset', type=str, default='resnet50',
                    choices=['resnet50', 'biomedclip'],
                    help='model architecture: resnet50 | biomedclip (default: biomedclip)')        
    
    return parser.parse_args()

def visualize_clusters(clinical_data, kmeans_labels, K):
    """
    Visualize clusters using a 2D scatter plot with a hollow circle style, larger markers, and transparency.
    """
    plt.figure(figsize=(8, 6))
    
    # Scatter plot with hollow circles ('o'), larger markers, and transparency (alpha)
    plt.scatter(clinical_data[:, 2], clinical_data[:, 6], 
                c=kmeans_labels, edgecolor='black', facecolors='none', 
                s=80, alpha=0.7, cmap='rainbow', marker='o')  # Adjusted marker size, transparency, and colormap
    
    plt.title(f'KMeans Clustering with K={K}') 
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)  # Added grid to make it more like the example screenshot
    plt.savefig(f'KMeans_Clustering_with_K={K}.png')
    plt.show()

import networkx as nx
import matplotlib.pyplot as plt

def visualize_adjacency_matrix(adj_matrix, feature_type_idx, node_labels=None):
    """
    Visualize a subgraph of the adjacency matrix with only a specified number of nodes (default 4).
    """
    num_nodes = 6
     # Limit the graph to only the first 'num_nodes' nodes
    sub_adj_matrix = adj_matrix[:num_nodes, :num_nodes]
    
    # Create a graph from the sub-adjacency matrix using the updated function
    G = nx.from_numpy_array(sub_adj_matrix)
    
    # Set up the plot
    plt.figure(figsize=(8, 8))
    
    # Use a spring layout for better visualization of the graph
    pos = nx.spring_layout(G)  # Omit random_state if not supported
    
    # Draw the graph with nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
            node_size=1000, edge_color='gray', font_size=10, font_weight='bold')
    
    # Add labels to nodes if provided
    if node_labels is not None:
        # Ensure we only label the subgraph's nodes
        labels = {i: node_labels[i] for i in range(num_nodes)}
        nx.draw_networkx_labels(G, pos, labels=labels)
            
    plt.title(f'SubGraph Representation of Adjacency Matrix for Feature Type {feature_type_idx}')
    plt.savefig(f'Graph Adjacency Matrix Type {feature_type_idx}.png')
    plt.show()

def main():
    args = parse_args()
    K = args.K

    # ----------------------------------------------
    # ------------------- Images -------------------
    # ----------------------------------------------

    path_ = 'dataset/extracted_feature/'
    train_feature = np.loadtxt(path_ + 'train_feature.csv',delimiter=',',dtype=np.float32)
    valid_feature = np.loadtxt(path_ + 'val_feature.csv',delimiter=',',dtype=np.float32)
    test_feature = np.loadtxt(path_ + 'test_feature.csv',delimiter=',',dtype=np.float32)

    train_id = pd.read_csv(path_ +'train_id.csv', header=None, names=['user_id'], dtype=str)
    valid_id = pd.read_csv(path_ +'val_id.csv', header=None, names=['user_id'], dtype=str)
    test_id = pd.read_csv(path_ +'test_id.csv', header=None, names=['user_id'], dtype=str)

    id_list = []
    image_list = []
    for i in range(len(train_id)):
        a = str(train_id['user_id'][i])
        id_list.append(a)
        image_list.append(train_feature[i])

    for i in range(len(valid_id)):
        a = str(valid_id['user_id'][i])
        id_list.append(a)
        image_list.append(valid_feature[i])
        
    for i in range(len(test_id)):
        a = str(test_id['user_id'][i])
        id_list.append(a)
        image_list.append(test_feature[i])

    # ------------------------------------------------
    # ------------------- Clinical -------------------
    # ------------------------------------------------

    dataset_features_categorical = pd.read_csv('dataset/data_summary.csv')
    dataset_features,category_mappings = convert_categorical_to_numeric(dataset_features_categorical)
    dataset_features['user_id'] = dataset_features['filename'].str.extract(r'(\d+)')

    # Split the dataset based on the 'use' column
    train_features_clinical = dataset_features[dataset_features['use'] == 'training']
    test_features_clinical  = dataset_features[dataset_features['use'] == 'test']
    val_features_clinical  = dataset_features[dataset_features['use'] == 'validation']

    # Sorting train, test, and val features
    train_clinical_sorted = pd.merge(train_id, train_features_clinical, on="user_id", how='inner')
    test_clinical_sorted = pd.merge(test_id, test_features_clinical, on="user_id", how='inner')
    val_clinical_sorted = pd.merge(valid_id, val_features_clinical, on="user_id", how='inner')

    # Concat the data in the same order of images
    clinical_data = pd.concat([train_clinical_sorted,val_clinical_sorted,test_clinical_sorted])
    clinical = clinical_data[['age' ,'gender' ,'race' ,'ethnicity' ,'language' ,'maritalstatus']]
    # text_features = clinical_data[['note',	'gpt4_summary']]
    labels = clinical_data['glaucoma']
    # image_ids = clinical_data['user_id']

    # features_names = ['female', 'male', 'race_asian', 'race_black', 'race_white',
    #                 'ethnicity_hispanic', 'ethnicity_non-hispanic', 'ethnicity_unknown',
    #                 'language_english', 'language_other', 'language_spanish', 'language_unknown',
    #                 'maritalstatus_divorced', 'maritalstatus_legally separated', 'maritalstatus_married or partnered', 'maritalstatus_single', 'maritalstatus_unknown', 'maritalstatus_widowed',
    #                 'age']
    features_names = ['gender' ,'race' ,'ethnicity' ,'language' ,'maritalstatus','age']
    # Getting train,valid,test indexes
    indexes = [i for i in range(len(labels))]

    train_index = indexes[:len(train_clinical_sorted)]
    valid_index = indexes[len(train_clinical_sorted):len(train_clinical_sorted)+len(val_clinical_sorted)]
    test_index = indexes[len(train_clinical_sorted)+len(val_clinical_sorted):]

    train_index = np.array(train_index)
    valid_index = np.array(valid_index)
    test_index = np.array(test_index)

    # Labels arrangement - Each row in y represents a label in a two-column format:
    # If the label is 0, the row will be [1, 0].
    # If the label is 1, the row will be [0, 1].
    labels = labels.to_list()
    y = np.zeros((len(labels),2))
    for i in range(len(labels)):
        if labels[i] == 0:
            y[i, 0] = 1
        else:
            y[i, 1] = 1

    # ---------------------------------------------------------------------------------------
    # ------------------------- Process: scale and transform to cat -------------------------
    # ---------------------------------------------------------------------------------------
    clinical = preprocess_cat_numeric(clinical)
    clinical = clinical[:, :-1] 
    # ---------------------------------------------------------------------------------------
    # ------------------ Concatenate Image features and non-image features ------------------
    # ---------------------------------------------------------------------------------------
    concat_feature = [] 
    for i in range(len(image_list)): #9952 - each row is a patient 
        concat = np.concatenate((np.expand_dims(image_list[i],axis=0),np.expand_dims(clinical[i],axis=0)),axis=1)
        concat_feature.append(concat[0])

    concate_feature_num = np.array(concat_feature)

    # --------------------------------------------------------------------------------------
    # ------------------------------------- Clustering -------------------------------------
    # --------------------------------------------------------------------------------------
    
    kmeans = KMeans(n_clusters=K, random_state = args.rnd)
    kmeans.fit(clinical.T)
    type_list = [[] for _ in range(K)]
    for i in range(len(clinical.T)):
        type_list[kmeans.labels_[i]].append(i)
    print(type_list)
    print('\n')
    
    from kmodes.kprototypes import KPrototypes # pip install kmodes

    # # Initialize K-Prototypes with the number of clusters and random state
    # kprototypes = KPrototypes(n_clusters=K, random_state=args.rnd)

    # # Fit the model on the clinical data transpose (assuming mixed data types)
    # kprototypes.fit(clinical.T, categorical=[i for i in range(1,18,1)])  # Specify the indices of categorical columns in `categorical`

    # # Create the type_list and assign points to clusters
    # type_list = [[] for _ in range(K)]
    # for i in range(len(clinical.T)):
    #     type_list[kprototypes.labels_[i]].append(i)
    
    # [['female', 'male'],
    #              ['race_asian', 'race_black', 'race_white','ethnicity_hispanic', 'ethnicity_non-hispanic', 'ethnicity_unknown','language_english', 'language_other', 'language_spanish', 'language_unknown'],
    #              ['maritalstatus_divorced', 'maritalstatus_legally separated', 'maritalstatus_married or partnered', 'maritalstatus_single', 'maritalstatus_unknown', 'maritalstatus_widowed'],
    #              ['age']]
    
    # type_list = [[0,2],[1,3,4,5,6,7,8,9,10,11],[12,13,14,15,16,17],[18]]
    # type_list = [[0,2,4],[1,3,5],[6]] # when dealing with categorical not one hot encoded

    # Use the clusters (kprototypes.labels_) as labels for classification evaluation
    # proto_clusters = kprototypes.labels_ 

    # visualize_clusters(clinical.T, kprototypes.labels_, K)
    # Create type_names by mapping indices to feature names
    type_names = [[features_names[i] for i in indices] for indices in type_list]
    # print(type_list) 
    # print(clinical[:,type_list[0]])
    
    # type_heatmap(clinical[:,type_list[0]].T,0,type_names[0])
    # type_heatmap(clinical[:,type_list[1]].T,1,type_names[1])
    # type_heatmap(clinical[:,type_list[2]].T,2,type_names[2])
    # type_heatmap(clinical[:,type_list[3]].T,3,type_names[3])    
    
    print('Clustering done:')
    for idx, names in enumerate(type_names):
        print(f"Type {idx + 1} names: {names}")
    print('\n')

    # -----------------------------------------------------------------------------------
    # ----------------------------- Creating Adjacency Matrix -----------------------------
    # -----------------------------------------------------------------------------------
    threses = args.thres.split(',')
    threses = [float(th) for th in threses] 
    save_list= [] #  list to store the adjacency matrices for different feature types

    # Processing Each Feature Type:
    k = 0 # counter
    age_column_index = 5  # Define the index of the 'age' feature

    for types in type_list:  # Iterate over different sets of features (clusters)
        print('type' + str(k))
        print("********")

        clinical_type = clinical[:, types]  # Extract features based on the current type list
        clinical_type = np.array(clinical_type)

        if age_column_index in types:  # Check if 'age' column (index 18) is in the current type list
            # Separate binary (categorical) and continuous (age) features
            binary_features = clinical_type[:, :-1]  # All columns except the last (binary vectors)
            age_feature = clinical_type[:, -1].reshape(-1, 1)  # The last column (continuous age feature)

            # Apply minmax scaling only to the continuous feature
            age_scaled = minmax_scale(age_feature)

            # Combine the one-hot encoded features with the scaled continuous feature
            p_ = np.hstack([binary_features, age_scaled])

            # Compute cosine similarity for the continuous feature (age)
            continuous_features = p_[:, -1].reshape(-1, 1)  # Extract continuous feature
            cos_sim_continuous = cosine_similarity(continuous_features)
            # cos_dis_continuous = 1 - cos_sim_continuous

            # Compute Jaccard similarity for the categorical features
            if len(types)>1:
                one_hot_features = p_[:, :-1]  # Extract one-hot encoded features (binary)
                jac_dis_categorical = pairwise_distances(one_hot_features, metric='jaccard')
                jac_sim_categorical = 1 - jac_dis_categorical
                # Combine Jaccard and Cosine distances with a weight alpha
                alpha = 0.7
                combined_sim = alpha * cos_sim_continuous + (1 - alpha) * jac_sim_categorical
            else:
                combined_sim = cos_sim_continuous

        else:  # If the 'age' feature is not part of the type list, handle all features as categorical
            # Compute Jaccard similarity for all categorical features
            jac_dis_categorical = pairwise_distances(clinical_type, metric='jaccard')
            jac_sim_categorical = 1 - jac_dis_categorical
            combined_sim = jac_sim_categorical  # Only use Jaccard distance


                
        print('combined_sim:')    
        print(combined_sim)

        # Now, construct adjacency matrix based on cosine and/or Jaccard distances
        adj_ = np.zeros(combined_sim.shape)
        thres = threses[k]  # Threshold for current type

        # Initialize binary adjacency matrix by applying the threshold
        adj_ = (combined_sim >= thres).astype(int)

        # Ensure diagonal elements are set to 1
        np.fill_diagonal(adj_, 1)        
        
        # Append the resulting adjacency matrix to the save list
        save_list.append(adj_)
        # visualize_adjacency_matrix(adj_, k)  
        print(adj_)      
        k += 1

    # --------------------------------------------------------------------------------------------------------
    # ------------------------------ Creating and saving the <FairClip> pkl file -----------------------------
    # --------------------------------------------------------------------------------------------------------
    # saving a multiplex network structure with adjacency matrices and relevant data into a Python pickle file.

    if K ==6:
        adj_0 = save_list[0]
        adj_1 = save_list[1]
        adj_2 = save_list[2]
        adj_3 = save_list[3]
        adj_4 = save_list[4]
        adj_5 = save_list[5]
        multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'type4':adj_4,'type5':adj_5,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}
    elif K ==5:
        adj_0 = save_list[0]
        adj_1 = save_list[1]
        adj_2 = save_list[2]
        adj_3 = save_list[3]
        adj_4 = save_list[4]
        multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'type4':adj_4,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}
    elif K ==4:
        adj_0 = save_list[0]
        adj_1 = save_list[1]
        adj_2 = save_list[2]
        adj_3 = save_list[3]
        multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}
    elif K ==3:
        adj_0 = save_list[0]
        adj_1 = save_list[1]
        adj_2 = save_list[2]
        multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}
    elif K ==2:
        adj_0 = save_list[0]
        adj_1 = save_list[1]
        multi = {'label':y,'type0':adj_0,'type1':adj_1,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature_num}


    with open(f'MultiplexNetwork/data/FairClip_{args.dataset}_categ.pkl', 'wb') as f:
        pickle.dump(multi, f, pickle.HIGHEST_PROTOCOL)
        
    print(f'FairClip_{args.dataset}_categ.pkl file is saved to MuliplexNetwork/data')
    
    #------------------------------------------------------------------------------------------
    #----------------------- create cluster labels based on image data ------------------------
    #------------------------------------------------------------------------------------------
    
    # Combine all features before clustering
    combined_features = np.vstack([train_feature, valid_feature, test_feature])
    # Perform K-Means clustering on the combined dataset
    kmeans = KMeans(n_clusters=args.K_images, random_state=0).fit(combined_features)

    # Get cluster labels for the combined dataset
    combined_cluster_labels = kmeans.labels_
    np.savetxt(f'MultiplexNetwork/data/cluster_labels_{args.dataset}_categ.csv', np.array(combined_cluster_labels, dtype=int), delimiter=',')

    print(f'cluster_labels_{args.dataset}_categ.csv file is saved to MuliplexNetwork/data')

if __name__ == '__main__':
    main()

# python Preprocessing/Non_image_preprocessing/kmeans.py --K 4