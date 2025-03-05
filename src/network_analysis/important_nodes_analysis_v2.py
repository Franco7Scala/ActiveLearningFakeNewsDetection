import os.path

import networkx as nx
import torch
import pandas as pd
from tqdm import tqdm
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.louvain import louvain_communities
from torch_geometric.utils import to_networkx
from networkx.algorithms.community import greedy_modularity_communities, asyn_lpa_communities

#from data_preprocessing.mumin import load_mumin_heterodata
from src.data_loading.politifact import load_politifact_heterodata
from src.data_utils import open_pickle, get_base_dir, save_dict_to_pickle
#from src.network_analysis.plot_users import user_stats


def construct_user_graph(hetero_data):
    for node_type in hetero_data.x_dict:
        hetero_data[node_type].node_id = torch.arange(hetero_data[node_type].x.size(0))
    nx_graph = to_networkx(hetero_data, to_undirected=False, node_attrs=['node_id'], edge_attrs=[])

    UU = set()

    for edge in nx_graph.edges(data=True):
        src, dst, edge_data = edge
        src_type = nx_graph.nodes[src].get('type')
        dst_type = nx_graph.nodes[dst].get('type')
        if src_type == 'user' and dst_type == 'user':
            UU.add((src, dst))

    UU = list(UU)

    uu_ret_tensor = hetero_data.edge_index_dict[('user', 'metapath_0', 'user')]
    uu_ret = list(set(map(tuple, uu_ret_tensor.t().tolist())))

    UU = list(set(UU + uu_ret))

    user_id_map = {node: nx_graph.nodes[node]['node_id'] for node in nx_graph.nodes if nx_graph.nodes[node]['type'] == 'user'}

    user_graph = nx.DiGraph()
    user_graph.add_edges_from(UU)

    original_user_ids = set(hetero_data['user'].node_id.tolist())
    for user_id in original_user_ids:
        if user_id not in user_id_map.values():
            user_graph.add_node(user_id)

    return user_graph, user_id_map




def get_topk_centrality_nodes(G, user_id_map, k, centrality_measure):
    if centrality_measure == "degree":
        centrality = nx.degree_centrality(G)
    elif centrality_measure == "closeness":
        centrality = nx.closeness_centrality(G)
    elif centrality_measure == "betweenness":
        centrality = nx.betweenness_centrality(G)
    elif centrality_measure == "eigenvector":
        centrality = nx.eigenvector_centrality(G)
    elif centrality_measure == "pagerank":
        centrality = nx.pagerank(G)
    elif centrality_measure == "katz":
        centrality = nx.katz_centrality(G)
    elif centrality_measure == "hits":
        centrality = nx.hits(G) #hubs, authorities = ...

    #user_nodes = {n: c for n, c in centrality.items()} # if nx_graph.nodes[n].get('type') == 'user'}
    user_nodes = {user_id: centrality.get(node, 0.0) for node, user_id in user_id_map.items()}
    #sorted_user_nodes = sorted(user_nodes.items(), key=lambda x: x[1], reverse=True)
    #top_k = sorted_user_nodes[:k]
    #return top_k
    return user_nodes



def community_detection_louvain(user_graph, user_id_map, hetero_data):
    original_user_ids = set(hetero_data['user'].node_id.tolist())
    for node in user_graph.nodes:
        if node not in user_id_map:
            user_id_map[node] = node
    connected_nodes = [node for node in user_graph.nodes if user_id_map[node] in original_user_ids]
    filtered_user_graph = user_graph.subgraph(connected_nodes)
    communities = louvain_communities(filtered_user_graph, seed=42)
    user_communities = {}
    for community_id, community_nodes in enumerate(communities):
        for node in community_nodes:
            mapped_id = user_id_map[node]
            user_communities[mapped_id] = community_id
    df = pd.DataFrame(list(user_communities.items()), columns=['user_id', 'community'])
    return df

#  return a dictionary where each user maps to a tuple (true_claims, false_claims)
def user_news_discussion_stats(hetero_data, target_type):
    news_to_tweet = hetero_data.edge_index_dict[(target_type, 'is_discussed_by', 'tweet')]
    tweet_to_user = hetero_data.edge_index_dict[('tweet', 'is_posted_by', 'user')]
    retweet_to_user = hetero_data.edge_index_dict[('tweet', 'is_retweeted_by', 'user')]
    tweet_to_user_combined = torch.cat([tweet_to_user, retweet_to_user], dim=1)

    news_labels = hetero_data[target_type].y

    num_users = hetero_data["user"].num_nodes
    user_stats = {user: {'true': 0, 'false': 0} for user in range(num_users)}


    for i in range(news_to_tweet.size(1)):
        news, tweet = news_to_tweet[:, i]
        news_label = news_labels[news].item()

        user_indices = torch.where(tweet_to_user_combined[0] == tweet)[0]
        users = tweet_to_user_combined[1, user_indices]

        for user in users.tolist():
            if user not in user_stats:
                user_stats[user] = {'true': 0, 'false': 0}
            if news_label == 0:
                user_stats[user]['true'] += 1
            elif news_label == 1:
                user_stats[user]['false'] += 1

    #user_stats_summary = {user: (stats['true'], stats['false']) for user, stats in user_stats.items()}
    #return user_stats_summary

    user_stats_summary = {user: (stats['true'], stats['false'], stats['true'] + stats['false']) for user, stats in user_stats.items()}

    sorted_users = sorted(user_stats_summary.items(), key=lambda x: x[1][2], reverse=True)


    return sorted_users


def check_important_nodes_in_topk(important_nodes_id, topk_users_deg, topk_users_bet, topk_users_clo, topk_users_pag):
    centrality_measures = {
        "degree_centrality": topk_users_deg,
        "betweenness_centrality": topk_users_bet,
        "closeness_centrality": topk_users_clo,
        "pagerank": topk_users_pag
    }

    overlap_results = {}

    for measure, topk_nodes in centrality_measures.items():
        # Calculate the intersection of important nodes and top-k nodes
        overlap = important_nodes_id.intersection(set(topk_nodes))
        # Calculate the proportion of important nodes in the top-k nodes
        proportion = len(overlap) / len(important_nodes_id)
        overlap_results[measure] = proportion

    return overlap_results

def check_nodes_higher_discussions(user_ids, user_stats):
    users =[user for user, (true_claims, false_claims, total_claims) in user_stats]
    return user_ids.intersection(set(users))


def check_nodes_position(central_nodes, important_nodes, no_of_parts):
    # Convert large_set to a list for positional indexing
    important_nodes = list(important_nodes)
    large_size = len(important_nodes)

    part_size = large_size // no_of_parts

    sections = [
        set(important_nodes[i * part_size:(i + 1) * part_size])
        for i in range(no_of_parts)
    ]

    section_counts = [sum(1 for node in central_nodes if node in section) for section in sections]

    # Find the index of the dominant section
    dominant_section_index = section_counts.index(max(section_counts))

    # Return results
    return dominant_section_index, section_counts

def topk_communities_analysis(df_communities, top_k_users, strategy):
    topk_communities = df_communities[df_communities['user_id'].isin(top_k_users)] #include only rows with user_id in top_k_users
    community_counts = topk_communities.groupby('community_id').size().reset_index(name='topk_user_count') #group by community_id and count the number of top-k users in each community
    num_communities_with_topk = community_counts.shape[0] #compute the number of communities containing at least one top-k node
    communities_with_multiple_topk = community_counts[community_counts['topk_user_count'] > 1] #identify communities containing multiple top-k nodes
    print(f"Total number of communities containing at least one top-k node: {num_communities_with_topk}")
    print(f"Number of communities containing multiple top-k nodes: {communities_with_multiple_topk.shape[0]}")
    community_counts.to_csv(os.path.join('out_dir', f'community_counts_{strategy}.csv'), index=False)
    communities_with_multiple_topk.to_csv(os.path.join('out_dir', f'communities_with_multiple_topk_{strategy}.csv'), index=False)





""" LOAD/GENERATE GRAPHS """
#print("Graph generation...")
heterodata = load_politifact_heterodata()
#user_graph, user_id_map = construct_user_graph(heterodata)
#print(user_graph)


""" COMPUTE USER STATS wrt LABELS"""
user_stats_labels = user_news_discussion_stats(heterodata, target_type="news")
df = pd.DataFrame(
    [(user_id, inner_tuple[0], inner_tuple[1], inner_tuple[2]) for user_id, inner_tuple in user_stats_labels],
    columns=["user_id", "no_true_news_discussed", "no_false_news_discussed", "no_tot_news_discussed"]
)
df.to_csv(os.path.join(get_base_dir(), "users_stats_labels.csv"), index=False)
print(df.shape)


#out_dir = get_base_dir()
'''
fname = "user_label_stats.pkl"
save_dict_to_pickle(user_stats, os.path.join(out_dir, fname))
'''

""" COMMUNITY DETECTION """

'''
print("Performing community detection...")
df = community_detection_louvain(user_graph, user_id_map, heterodata)
df.to_csv(os.path.join(out_dir, 'user_communities_louvain.csv'), index=False)

df = pd.read_csv(os.path.join(out_dir, 'user_communities_louvain.csv'))
num_communities = df['community'].nunique()
print(f"Number of communities detected: {num_communities} for {df.shape[0]} users")
'''

""" COMPUTE CENTRALITY MEASURES AND TOP-k NODES"""

'''
print("processing degree centrality...")
topk_degree = get_topk_centrality_nodes(user_graph, user_id_map, k, "degree")
df_degree = pd.DataFrame(list(topk_degree.items()), columns=['user_id', 'degree_centrality_score'])
df_degree.to_csv(os.path.join(out_dir, 'user_degree_centrality.csv'), index=False)
print(df_degree.head())

print("processing pagerank...")
topk_pagerank = get_topk_centrality_nodes(user_graph, user_id_map, k, "pagerank")
df_pagerank = pd.DataFrame(list(topk_pagerank.items()), columns=['user_id', 'pagerank_score'])
df_pagerank.to_csv(os.path.join(out_dir, 'user_pagerank.csv'), index=False)
print(df_pagerank.head())

print("processing betweenness centrality...")
topk_betweenness = get_topk_centrality_nodes(user_graph, user_id_map, 50, "betweenness")
df_betweenness = pd.DataFrame(list(topk_betweenness.items()), columns=['user_id', 'betweenness_centrality_score'])
df_betweenness.to_csv(os.path.join(out_dir, 'user_betweenness_centrality.csv'), index=False)
print(df_betweenness.head())

print("processing closeness centrality...")
topk_closeness = get_topk_centrality_nodes(user_graph, user_id_map, 50, "closeness")
df_closeness = pd.DataFrame(list(topk_closeness.items()), columns=['user_id', 'closeness_centrality_score'])
df_closeness.to_csv(os.path.join(out_dir, 'user_closeness_centrality.csv'), index=False)
print(df_closeness.head())
'''
'''
k = 10000 # int(len(all_users)*0.1) #100, 1000

df_deg = pd.read_csv(os.path.join(out_dir, "network_analysis", 'user_degree_centrality.csv'))
top_k_users_deg = set(df_deg.nlargest(k, 'degree_centrality_score')['user_id'])

df_pag = pd.read_csv(os.path.join(out_dir, "network_analysis", 'user_pagerank.csv'))
top_k_users_pag = set(df_pag.nlargest(k, 'pagerank_score')['user_id'])

df_bet = pd.read_csv(os.path.join(out_dir, "network_analysis", 'user_betweenness_centrality.csv'))
top_k_users_bet = set(df_bet.nlargest(k, 'betweenness_centrality_score')['user_id'])

df_clo = pd.read_csv(os.path.join(out_dir, "network_analysis", 'user_closeness_centrality.csv'))
top_k_users_clo = set(df_clo.nlargest(k, 'closeness_centrality_score')['user_id'])

common_top_k = top_k_users_deg & top_k_users_pag & top_k_users_bet & top_k_users_clo
print(f"Users common across all top k sets: {len(common_top_k)}")

#important_node_ids = open_pickle(os.path.join(get_base_dir(), "selected_1_0_margin.pkl")) #last 100: [-100:]
all_users = open_pickle(os.path.join(get_base_dir(), "selected_1_0_margin_politifact.pkl"))["user"]
print(f"No. of users: {len(all_users)}")
node_ids_AL_margin = set(open_pickle(os.path.join(get_base_dir(), "selected_1_0_margin_politifact.pkl"))["user"][:k]) #topo k: [:k], last k: [-k:]
node_ids_AL_entropy = set(open_pickle(os.path.join(get_base_dir(), "selected_1_0_entropy_politifact.pkl"))["user"][:k])
node_ids_lc = set(open_pickle(os.path.join(get_base_dir(), "selected_1_0_leastconfidence_politifact.pkl"))["user"][:k])
common_top_k_AL = node_ids_AL_entropy & node_ids_AL_margin & node_ids_AL_margin
print(f"Users common across all top k sets (AL): {len(common_top_k_AL)}")
'''



'''
res = check_important_nodes_in_topk(important_node_ids, top_k_users_deg, top_k_users_bet, top_k_users_clo, top_k_users_pag)
for measure, proportion in res.items():
    print(f"Proportion of important nodes in top-k {measure}: {proportion:.2f}")

test = check_nodes_higher_discussions(important_node_ids, user_stats)
print(len(test))
'''

'''
no_of_parts = 100

dominant_section_deg, sections_deg = check_nodes_position(top_k_users_deg, important_node_ids, no_of_parts)
print("degree centrality")
print(dominant_section_deg)
print(sections_deg)
print()

dominant_section_bet, sections_bet = check_nodes_position(top_k_users_bet, important_node_ids, no_of_parts)
print("betweenness centrality")
print(dominant_section_bet)
print(sections_bet)
print()

dominant_section_clo, sections_clo = check_nodes_position(top_k_users_clo, important_node_ids, no_of_parts)
print("closeness centrality")
print(dominant_section_clo)
print(sections_clo)
print()

dominant_section_pag, sections_pag = check_nodes_position(top_k_users_pag, important_node_ids, no_of_parts)
print("pagerank")
print(dominant_section_pag)
print(sections_pag)
print()

#topk_communities_analysis(df, top_k_users_deg, "degree_centrality") #df_communities
#topk_communities_analysis(df, top_k_users_bet, "betweenneess_centrality") #df_communities
#topk_communities_analysis(df, top_k_users_clo, "closeness_centrality") #df_communities
#topk_communities_analysis(df, top_k_users_pag, "pagerank") #df_communities

#topk_communities_analysis(df, important_node_ids, "AL_margin") #df_communities
'''