import os
import pandas as pd
import torch
import networkx as nx
from torch_geometric.utils import to_networkx

from src.data_preprocessing.politifact import load_politifact_heterodata
from src.data_utils import get_base_dir, save_dict_to_pickle


edge_types = {
        'mention': ('user', 'mentions', 'user'),
        'retweet': ('user', 'metapath_0', 'user') #,
        #'same_news': ('user', 'metapath_1', 'user'),
        #'same_hashtag': ('user', 'metapath_2', 'user')
}



def construct_users_edgelist(hetero_data, edge_type):
    src_list = []
    tgt_list = []
    weight_list = []

    if edge_type == 'mention':
        mention_edge_index = hetero_data[edge_types['mention']]['edge_index']
        src_list.extend(mention_edge_index[0].tolist())
        tgt_list.extend(mention_edge_index[1].tolist())
        weight_list = [1]*len(src_list)

    elif edge_type == 'retweet':
        retweet_edge_index = hetero_data[edge_types['retweet']]['edge_index']
        retweet_edge_weight = hetero_data[edge_types['retweet']]['edge_weight']
        src_list.extend(retweet_edge_index[0].tolist())
        tgt_list.extend(retweet_edge_index[1].tolist())
        weight_list.extend(retweet_edge_weight.tolist())

    elif edge_type == 'same_news':
        same_news_edge_index = hetero_data[edge_types['same_news']]['edge_index']
        same_news_edge_weight = hetero_data[edge_types['same_news']]['edge_weight']
        src_list.extend(same_news_edge_index[0].tolist())
        tgt_list.extend(same_news_edge_index[1].tolist())
        weight_list.extend(same_news_edge_weight.tolist())
        src_list.extend(same_news_edge_index[1].tolist())  # Reverse direction
        tgt_list.extend(same_news_edge_index[0].tolist())
        weight_list.extend(same_news_edge_weight.tolist())

    elif edge_type == 'same_hashtag':
        same_hashtag_edge_index = hetero_data[edge_types['same_hashtag']]['edge_index']
        same_hashtag_edge_weight = hetero_data[edge_types['same_hashtag']]['edge_weight']
        src_list.extend(same_hashtag_edge_index[0].tolist())
        tgt_list.extend(same_hashtag_edge_index[1].tolist())
        weight_list.extend(same_hashtag_edge_weight.tolist())
        src_list.extend(same_hashtag_edge_index[1].tolist())  # Reverse direction
        tgt_list.extend(same_hashtag_edge_index[0].tolist())
        weight_list.extend(same_hashtag_edge_weight.tolist())

    else:
        raise ValueError(f"Invalid edge type: {edge_type}. Valid types are {list(edge_types.keys())} or None for all.")

    df_edges = pd.DataFrame({
            'src': src_list,
            'tgt': tgt_list,
            'weight': weight_list
    })

    return df_edges


base_dir = get_base_dir()
"""
heterodata = load_politifact_heterodata()
print("heterodata loaded")

for edge_type in edge_types:
    print(f"Processing edge type: {edge_type}")
    df = construct_users_edgelist(heterodata, edge_type=edge_type)
    print(f"{edge_type} edges: {df.shape}")
    df.to_csv(os.path.join(base_dir, f'{edge_type}_edges.csv'), index=False)
"""
df = pd.read_csv(os.path.join(base_dir, 'original_data', 'nodes', 'user.csv'))
#df["weight"] = df["weight"].astype(int)
#df.to_csv(os.path.join(base_dir, 'retweet_edges.csv'), index=False)
print(df.shape)
