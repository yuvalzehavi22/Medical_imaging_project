import zipfile
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import os
import shutil
import argparse
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx



def parse_args():
    parser = argparse.ArgumentParser(description='kmeans')
    parser.add_argument('--K', type=int, default=3, help="Number of clusters (graphs)")
    parser.add_argument('--K_images', type=int, default=4, help="Number of clusters of image data ")
    parser.add_argument('--thres', type=str, default='0.95,0.91,0.93,0.9', help="Threshold value for similarity")
    parser.add_argument('--rnd', type=int, default=5, help="Random State for kmeans model")
    parser.add_argument('--PCA', action='store_false', default=False)
    parser.add_argument('--dataset', type=str, default='resnet50',
                    choices=['resnet50', 'biomedclip'],
                    help='model architecture: resnet50 | biomedclip (default: biomedclip)')    
   
    
    return parser.parse_args()

def visualize_clusters(transpose_num, kmeans,K):
    """
    Visualize clusters using a 2D scatter plot with a hollow circle style, larger markers, and transparency.
    """

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(transpose_num)

    # Get the predicted clusters from KMeans
    labels = kmeans.labels_

    # Plot the data with the cluster labels
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis', s=50)

    # Mark the cluster centers
    centers = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')

    plt.title('PCA: KMeans Clustering Visualization')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.colorbar()
    plt.savefig(f'KMeans_Clustering_with_K={K}.png')

def visualize_adjacency_matrix(adj_matrix, feature_type_idx, node_labels=None):
    """
    Visualize a subgraph of the adjacency matrix with only a specified number of nodes (default 4).
    """
    num_nodes = 4
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

    data_summary = pd.read_csv('dataset/data_summary.csv')
    clinic_note_sum = pd.read_csv('dataset/clinical_note_embeddings.csv')

    clinic_note_sum = clinic_note_sum.round(5)
    data_summary['user_id'] = data_summary['filename'].str.extract(r'(\d+)')

    without_age = False
    if without_age:
        data_summary.drop(columns = ['age', 'gender' ,'race' ,'ethnicity' ,'language' ,'maritalstatus', 'note', 'gpt4_summary', 'filename'],inplace=True)
        data_summary = data_summary[['user_id', 'use', 'glaucoma']]
    else:
        data_summary.drop(columns = ['gender' ,'race' ,'ethnicity' ,'language' ,'maritalstatus', 'note', 'gpt4_summary', 'filename'],inplace=True)
        data_summary = data_summary[['user_id', 'use', 'glaucoma', 'age']]


    dataset_features = pd.concat([data_summary, clinic_note_sum], axis=1)

    # Split the dataset based on the 'use' column
    train_features_clinical = dataset_features[dataset_features['use'] == 'training']
    test_features_clinical  = dataset_features[dataset_features['use'] == 'test']
    val_features_clinical  = dataset_features[dataset_features['use'] == 'validation']

    # Sorting train, test, and val features
    train_clinical_sorted = pd.merge(train_id, train_features_clinical, on="user_id", how='inner')
    test_clinical_sorted = pd.merge(test_id, test_features_clinical, on="user_id", how='inner')
    val_clinical_sorted = pd.merge(valid_id, val_features_clinical, on="user_id", how='inner')

    # Concat the data in the same order of images
    clinical_data = pd.concat([train_clinical_sorted,val_clinical_sorted,test_clinical_sorted], ignore_index=True)

    clinical_data['glaucoma'] = clinical_data['glaucoma'].map({'yes': 1, 'no': 0})
    labels = clinical_data['glaucoma']
    image_ids = clinical_data['user_id']

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
   # --------------------------------------------------------------------------------------
    # ------------------------------------- Clustering -------------------------------------
    # --------------------------------------------------------------------------------------
                
    use_col = clinical_data.keys().tolist()
    k_means_list = use_col[3:]
    clinical = clinical_data[k_means_list]

    use_clinical_minmax = minmax_scale(clinical_data[k_means_list], axis=0, copy =True)
    use_k_means = clinical_data.copy()
    use_k_means[k_means_list] = use_clinical_minmax

    transpose_num = use_k_means[k_means_list].T
    kmeans = KMeans(n_clusters=K, random_state = args.rnd)
    kmeans.fit(transpose_num)
    
    visualize_clusters(transpose_num, kmeans,K)
    
    type_list = [[] for _ in range(K)]
    for i in range(len(transpose_num)):
        type_list[kmeans.labels_[i]].append(i)


    # Create type_names by mapping indices to feature names
    type_names = [[k_means_list[i] for i in indices] for indices in type_list]
    print('Clustering done.')
    # for idx, names in enumerate(type_names):
    #     print(f"Type {idx + 1} names: {names}")
    print('\n')
    
    # -----------------------------------------------------------------------------------
    # ----------------------------- Creating Adjacency Matrix -----------------------------
    # -----------------------------------------------------------------------------------
    print('Creating adjacency matrices . . . ')

    threses = args.thres.split(',')
    threses = [float(th) for th in threses] 
    save_list= [] #  list to store the adjacency matrices for different feature types
    k=0
    for types in type_names:
        before_adj_ = []
        use_clinical_dummy_multi = clinical_data[['glaucoma']+types]
        ll = use_clinical_dummy_multi.drop(columns=['glaucoma'])
        
        for id_ in id_list:
            lp = clinical_data[clinical_data['user_id']==id_].index.item()
            p = ll.loc[lp].tolist()
            before_adj_.append(p)
        p_ = np.array(before_adj_)
        # p_ = minmax_scale(p_, axis=0, copy =True)

        cos_ = sklearn.metrics.pairwise.cosine_similarity(p_,p_)

        # plt.figure()
        # plt.hist(cos_.flatten(), bins=50, color='blue', alpha=0.7)
        # plt.title(f"Histogram of combined_sim Matrix for type {k}")
        # plt.xlabel("Values")
        # plt.ylabel("Frequency")
        # plt.show()

        adj_ = np.zeros(cos_.shape)
        thres = threses[k]
        for i in range(cos_.shape[0]):
            for j in range(cos_.shape[0]):
                if cos_[i][j] > thres:
                    adj_[i][j] = 1
                else:
                    adj_[i][j] = 0
        save_list.append(adj_)
        visualize_adjacency_matrix(adj_, k)  
        k+=1
   
    # -----------------------------------------------------------------------------------
    # ----------------------------- PCA -------------------------------------------------
    # -----------------------------------------------------------------------------------
    # Assuming clinical_note_embeddings is your 768-dimensional embedding matrix
    if args.PCA:    
        pca = PCA(n_components=300)
        transpose_num = pca.fit_transform(transpose_num.T)
        print(f'Shape of reduced embeddings: {transpose_num.shape}')

    # ---------------------------------------------------------------------------------------
    # ------------------ Concatenate Image features and non-image features ------------------
    # ---------------------------------------------------------------------------------------
    concat_feature = [] 
    for i in range(len(image_list)):
        concat = np.concatenate((np.expand_dims(image_list[i],axis=0),np.expand_dims(transpose_num[i],axis=0)),axis=1)
        concat_feature.append(concat[0])

    concate_feature_num = np.array(concat_feature)
    print(f'Shape of concatenated data: {concate_feature_num.shape}')

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


    if args.PCA:
        IS_PCA = 'PCA'
    else:
        IS_PCA = 'without_PCA'
            
    with open(f'MultiplexNetwork/data/FairClip_{args.dataset}_text.pkl', 'wb') as f:
        pickle.dump(multi, f, pickle.HIGHEST_PROTOCOL)
        
    print(f'FairClip_{args.dataset}_text.pkl file saved at MuliplexNetwork/data')    
    
    #------------------------------------------------------------------------------------------
    #----------------------- create cluster labels based on image data ------------------------
    #------------------------------------------------------------------------------------------
    
    # Combine all features before clustering
    combined_features = np.vstack([train_feature, valid_feature, test_feature])
    # Perform K-Means clustering on the combined dataset
    kmeans = KMeans(n_clusters=args.K_images, random_state=0).fit(combined_features)

    # Get cluster labels for the combined dataset
    combined_cluster_labels = kmeans.labels_
    np.savetxt(f'MultiplexNetwork/data/cluster_labels_{args.dataset}.csv', np.array(combined_cluster_labels, dtype=int), delimiter=',')

    print('cluster_labels.csv file is saved to MuliplexNetwork/data')

if __name__ == '__main__':
    main()
