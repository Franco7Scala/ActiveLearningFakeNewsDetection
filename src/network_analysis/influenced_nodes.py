from collections import defaultdict
import torch
import os
from torch_geometric.data import HeteroData

from src.data_loading.mumin import load_mumin_heterodata
from src.data_loading.politifact_node_type import load_politifact_heterodata
from src.network_analysis.centrality_measures import extract_homogeneous_weighted_graph
from src.network_analysis.ranking_combination import merge_rankings
from src.utils import get_base_dir, load_from_pickle, default_dict_float, save_to_pickle, default_dict_dict


def count_influenced_users(data: HeteroData, ranking: list, k: int, max_steps: int) -> int:
    """
    Counts the number of influenced user nodes given a top-k ranking and max_steps of reachability.

    Parameters:
    - data (HeteroData): The heterogeneous graph data.
    - ranking (list): Ordered list of user node IDs based on some ranking.
    - k (int): Number of top-ranked users to consider as influencers.
    - max_steps (int): Maximum number of steps for reachability.

    Returns:
    - int: The number of influenced users not in the top-k ranking.
    """
    # Extract top-k ranked user nodes
    top_k_users = set(ranking[:k])

    # Extract directed edges where 'user' is both source and target
    user_edges = []
    for edge_type in data.edge_types:
        if edge_type[0] == 'user' and edge_type[2] == 'user':
            user_edges.append(data[edge_type].edge_index)

    # Merge all edges into a single adjacency list
    if not user_edges:
        return 0  # No user-to-user relations

    edge_index = torch.cat(user_edges, dim=1)  # Merge all edges

    # Perform BFS to find influenced nodes
    influenced_users = set()
    frontier = list(top_k_users)
    visited = set(frontier)

    for _ in range(max_steps):
        next_frontier = set()
        for user in frontier:
            # Get neighbors of the current user
            mask = edge_index[0] == user
            neighbors = edge_index[1][mask].tolist()
            for neighbor in neighbors:
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)

        # Update frontier with new reachable users
        if not next_frontier:
            break  # No more nodes to explore
        frontier = list(next_frontier)

        # Add newly discovered users (excluding top-k ones)
        influenced_users.update(next_frontier - top_k_users)

    return len(influenced_users)

def count_influenced_users_in_pair(data: HeteroData, ranking1: list, ranking2:list, k: int, max_steps: int) -> int:

    ranking = merge_rankings(ranking1, ranking2, k)

    # Extract top-k ranked user nodes
    top_k_users = set(ranking[:k])

    # Extract directed edges where 'user' is both source and target
    user_edges = []
    for edge_type in data.edge_types:
        if edge_type[0] == 'user' and edge_type[2] == 'user':
            user_edges.append(data[edge_type].edge_index)

    # Merge all edges into a single adjacency list
    if not user_edges:
        return 0  # No user-to-user relations

    edge_index = torch.cat(user_edges, dim=1)  # Merge all edges

    # Perform BFS to find influenced nodes
    influenced_users = set()
    frontier = list(top_k_users)
    visited = set(frontier)

    for _ in range(max_steps):
        next_frontier = set()
        for user in frontier:
            # Get neighbors of the current user
            mask = edge_index[0] == user
            neighbors = edge_index[1][mask].tolist()
            for neighbor in neighbors:
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)

        # Update frontier with new reachable users
        if not next_frontier:
            break  # No more nodes to explore
        frontier = list(next_frontier)

        # Add newly discovered users (excluding top-k ones)
        influenced_users.update(next_frontier - top_k_users)

    return len(influenced_users)


if __name__ == '__main__':

    tgt_type = "news"
    base_dir = get_base_dir()
    #base_dir = "/mnt/nas/martirano/mumin"
    node_type = "user"
    heterodata = load_politifact_heterodata(get_base_dir(), node_type)
    #heterodata = load_mumin_heterodata(base_dir, node_type)
    #nx_graph, ids_map = extract_homogeneous_weighted_graph(heterodata, node_type)
    #print(nx_graph)

    dir_base = os.path.join(base_dir, "node_ranking")

    gnn_expl_feats = load_from_pickle(os.path.join(dir_base, "selected_GNNExplainer.pkl"))[node_type]
    gnn_expl_rel = load_from_pickle(os.path.join(dir_base, "selected_GNNExplainer_by_relations.pkl"))[node_type]
    gnn_expl = [gnn_expl_feats, gnn_expl_rel]

    al_entropy = load_from_pickle(os.path.join(dir_base, f"selected_EntropyALTechnique_target_{tgt_type}_ranking_{node_type}.pkl"))[node_type]
    al_lcs = load_from_pickle(os.path.join(dir_base, f"selected_LCSALTechnique_target_{tgt_type}_ranking_{node_type}.pkl"))[node_type]
    al_margin = load_from_pickle(os.path.join(dir_base, f"selected_MarginALTechnique_target_{tgt_type}_ranking_{node_type}.pkl"))[node_type]

    al_techniques = [al_entropy, al_lcs, al_margin]
    post_hoc_techniques = gnn_expl + al_techniques
    post_hoc_techniques_names = ["GNN_expl_feats", "GNN_expl_rel", "AL_entropy", "AL_lcs", "AL_margin"]

    betwenness = load_from_pickle(os.path.join(dir_base, f"selected_betweenness_centrality_{node_type}.pkl"))
    closeness = load_from_pickle(os.path.join(dir_base, f"selected_closeness_centrality_{node_type}.pkl"))
    degree = load_from_pickle(os.path.join(dir_base, f"selected_degree_centrality_{node_type}.pkl"))
    pagerank = load_from_pickle(os.path.join(dir_base, f"selected_pagerank_centrality_{node_type}.pkl"))
    voterank = load_from_pickle(os.path.join(dir_base, f"selected_voterank_centrality_{node_type}.pkl"))
    centrality_measures = [degree, pagerank, voterank, betwenness, closeness]
    centrality_measures_names = ["degree", "pagerank", "voterank", "betweenness", "closeness"]
    #centrality_measures = [degree, pagerank, closeness]
    #centrality_measures_names = ["degree", "pagerank", "closeness"]

    rankings = post_hoc_techniques + centrality_measures
    names = post_hoc_techniques_names + centrality_measures_names


    #ret = defaultdict(default_dict_dict)  # FA schifo ma pickle non fa serializzare lambda function
    ret = {}

    MAX_STEPS = [2, 3]
    K = [2, 4, 8, 16, 32, 64]
    tot = len(al_margin)
    TOP_K = [tot // k + 1 for k in K]  # 1/2, 1/4, 1/8, 1/16

    for j, k in enumerate(TOP_K):
        ret[k] = {}
        for max_steps in MAX_STEPS:
            ret[k][max_steps] = {}
            for i, ranking in enumerate(rankings):
                cont_influenced = count_influenced_users(heterodata, ranking, k, max_steps)
                print(f"Processing {names[i]}")
                ret[k][max_steps][names[i]] = cont_influenced
                print(f"No. of reachable nodes by {names[i]} considering top-1/{K[j]} in max {max_steps} = {cont_influenced}")

            save_to_pickle(ret, os.path.join(dir_base, f"influenced_nodes_{node_type}_count_checkpoint.pkl")) #checkpoint

    save_to_pickle(ret, os.path.join(dir_base, f"influenced_nodes_{node_type}_count_all.pkl"))