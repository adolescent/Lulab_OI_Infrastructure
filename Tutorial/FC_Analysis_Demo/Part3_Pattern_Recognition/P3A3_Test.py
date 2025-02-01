'''
This script will test several graph methods for clustering.
methods are as below:

Louvain算法-鲁汶算法-不重叠社区，描述聚集化程度
Infomap 算法
COPRA 算法-可重叠
BigClam算法-可重叠


'''


#%% Load and import 

import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd
from Brain_Atlas.Atlas_Mask import Mask_Generator
from Atlas_Corr_Tools import *
from scipy.stats import pearsonr
from Signal_Functions.Pattern_Tools import Do_PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist,squareform
import time

wp = r'D:\_DataTemp\OIS\Wild_Type\Preprocessed'
MG = Mask_Generator(bin=4)

on_response,on_mask = cf.Load_Variable(wp,'Response_1d.pkl')



#%%
'''
Test 1, Lovian algoriths

'''
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import community.community_louvain as louvain # install as ,but use as community.

# calculate distance matrix, using pearson r.
normalized_ts = (on_response - on_response.mean(axis=1, keepdims=True))/on_response.std(axis=1, keepdims=True)
corr_matrix = np.corrcoef(normalized_ts)

# Convert correlations to distances (since Louvain uses similarities)
distance_matrix = 1 - corr_matrix  # 0 = identical, 2 = anti-correlated
np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0


#%% build the connetion graph

k = 30 # Number of neighbors per node. Bigger the neighbor number, globaller the graph.

# Create adjacency matrix using k-NN
# adjacency = kneighbors_graph(
#     distance_matrix,
#     n_neighbors=k,
#     mode='distance',
#     metric='precomputed',  # Use precomputed distances
#     include_self=False
# )
adjacency = kneighbors_graph(normalized_ts,n_neighbors=k,mode='connectivity',metric='l2',include_self=False)

# Convert distances to similarities (weights)
# adjacency.data = 1 / (1 + adjacency.data)  # Invert distance to similarity
adjacency = (adjacency + adjacency.T) / 2  # Symmetrize the matrix

# create networkx graph
G = nx.from_scipy_sparse_array(adjacency)

# actually method is only used for partition seperation, lovian only on this step.

partition = louvain.best_partition(G, weight='weight', resolution=1)# a dict of best partition.

community_df = pd.DataFrame({
    'pixel_id': partition.keys(),
    'community': partition.values()
}).sort_values(by='community')
# Group pixels by community
communities = community_df.groupby('community')['pixel_id'].apply(list).to_dict()
print(f"Found {len(communities)} communities.")

#%% visualization graph
pos = nx.spring_layout(G,seed=114514)  # Fruchterman-Reingold layout

# Color nodes by community
colors = [partition[node] for node in G.nodes()]

# Plot
fig,ax = plt.subplots(ncols=1,nrows=1,figsize = (8,6),dpi=180)



import matplotlib.patches as mpatches
# Create legend components
unique_groups = sorted(set(colors))
cmap = plt.cm.jet  # Use the same colormap as your nodes
norm = plt.Normalize(vmin=min(colors), vmax=max(colors))  # Same normalization

# Create legend patches
legend_patches = [
    mpatches.Patch(color=cmap(norm(group)), label=f'Group {group}')
    for group in unique_groups
]

# Add legend to plot
ax.legend(handles=legend_patches, title='Node Groups')
# plt.axis('off')

nx.draw_networkx_nodes(G, pos, node_size=5, node_color=colors,cmap='jet',ax = ax)
nx.draw_networkx_edges(G, pos, alpha=0.01,ax = ax)
plt.title("Pixel Communities (Louvain)")
plt.axis('off')

plt.show()



#%% visualization of clusts
c_vec = list(community_df.sort_values(by='pixel_id')['community']+1)

recover = copy.deepcopy(on_mask).astype('i4')
recover[on_mask==True] = c_vec
plt.imshow(recover,cmap='jet')




#%%
'''
Test 2, Infomap algorithm.
This algorithms allow nodes overlapping, and will return weight of each node to neighbor.

'''

from cdlib import algorithms
# from cdlib.utils import convert_graph_formats


num_communities = 7  # Example value for 1574 nodes

# Detect communities
communities = algorithms.infomap(G)

#%%
community_map = {}
for idx, community in enumerate(communities.communities):
    for node in community:
        community_map[node] = idx

# 5. Calculate node strengths (sum of edge weights)
node_strengths = np.zeros(corr_matrix.shape[0])
for node in G.nodes():
    node_strengths[node] = sum(
        G[node][neighbor]['weight'] 
        for neighbor in G.neighbors(node)
    )

from collections import defaultdict
community_strengths = defaultdict(list)
for node, strength in enumerate(node_strengths):
    community_strengths[community_map[node]].append(strength)


print("Community Analysis:")
for comm_id, strengths in community_strengths.items():
    print(f"Community {comm_id}:")
    print(f"  Number of nodes: {len(strengths)}")
    print(f"  Total strength: {sum(strengths):.2f}")
    print(f"  Average strength: {np.mean(strengths):.2f} ± {np.std(strengths):.2f}\n")

# 6. Visualize the graph
def visualize_communities(G, community_map):
    """Visualize graph with community coloring"""
    plt.figure(figsize=(15, 10))
    
    # Get positions using spring layout
    pos = nx.spring_layout(G, iterations=20, seed=114514)
    
    # Generate color map
    colors = [community_map[node] for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_size=20,
        cmap=plt.cm.tab20,
        node_color=colors
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, alpha=0.1,
        edge_color='gray'
    )
    
    plt.title("Network Community Structure")
    plt.axis('off')
    plt.show()

visualize_communities(G, community_map)

#%% show community
community_df = pd.DataFrame({
    'pixel_id': community_map.keys(),
    'community': community_map.values()
}).sort_values(by='community')


c_vec = list(community_df.sort_values(by='pixel_id')['community']+1)


recover = copy.deepcopy(on_mask).astype('i4')
recover[on_mask==True] = c_vec
plt.imshow(recover,cmap='jet')


#%%
'''
Step 3, try some overlapping method.
'''


communities = algorithms.slpa(G,t=50, r=0.1) # t as iteration,r as threshold

#%%
# 3. Create probability matrix
from matplotlib import cm
from collections import defaultdict

node_ids = list(G.nodes())
community_list = communities.communities

# Initialize probability matrix
prob_matrix = np.zeros((len(node_ids), len(community_list)))

# Populate matrix based on community membership
for comm_idx, comm in enumerate(community_list):
    for node in comm:
        node_idx = node_ids.index(node)
        prob_matrix[node_idx, comm_idx] = 1  # Binary membership

# Convert to probability distribution (normalize per node)
prob_matrix = prob_matrix / prob_matrix.sum(axis=1, keepdims=True)
prob_matrix = np.nan_to_num(prob_matrix)  # Handle isolated nodes

def visualize_overlapping_communities(G, communities):
    plt.figure(figsize=(14, 10))
    
    # Get node positions
    pos = nx.spring_layout(G, seed=42)
    
    # Create color map for communities
    community_colors = cm.tab20(np.linspace(0, 1, len(communities.communities)))
    
    # Calculate overlap counts
    overlap_counts = defaultdict(int)
    for comm in communities.communities:
        for node in comm:
            overlap_counts[node] += 1

    # Draw non-overlapping nodes first
    non_overlap = [n for n in G.nodes() if overlap_counts[n] == 1]
    overlap = [n for n in G.nodes() if overlap_counts[n] > 1]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='grey')
    
    # Draw non-overlapping nodes
    for node in non_overlap:
        comm_ids = [i for i, comm in enumerate(communities.communities) if node in comm]
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node],
            node_color=[community_colors[comm_ids[0]]],
            node_size=300, alpha=0.8
        )
    
    # Draw overlapping nodes with multiple markers
    for node in overlap:
        comm_ids = [i for i, comm in enumerate(communities.communities) if node in comm]
        
        # Draw multiple semi-transparent circles
        for i, comm_id in enumerate(comm_ids):
            nx.draw_networkx_nodes(
                G, pos, nodelist=[node],
                node_color=[community_colors[comm_id]],
                node_size=8+ 2*i,
                alpha=0.4,
                edgecolors='black',
                linewidths=1
            )
        
        # Add annotation for overlap count
        plt.text(
            pos[node][0], pos[node][1]+0.03,
            str(len(comm_ids)),
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=community_colors[i], markersize=10,
                  label=f'Community {i}')
        for i in range(len(communities.communities))
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("SLPA Community Detection with Overlapping Nodes Highlighted")
    plt.axis('off')
    plt.show()

# Run visualization
visualize_overlapping_communities(G, communities)

# Print probability matrix shape
print("\nProbability matrix shape:", prob_matrix.shape)
print("Example node probabilities (Node 0):", prob_matrix[0])

