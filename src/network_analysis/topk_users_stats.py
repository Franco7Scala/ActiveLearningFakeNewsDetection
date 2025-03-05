import pandas as pd
import os

from src.data_utils import get_base_dir, open_pickle, save_dict_to_pickle

base_dir = get_base_dir()
no_of_parts = 10
k_small_set = 100
#all_users =
#k_big_set = #int(len(all_users)*0.1) #100, 1000, 10000


def check_nodes_position(small_set, large_set, no_of_parts):
    # Convert large_set to a list for positional indexing
    all_nodes = list(large_set)
    large_size = len(all_nodes)

    part_size = large_size // no_of_parts

    sections = [
        set(all_nodes[i * part_size:(i + 1) * part_size])
        for i in range(no_of_parts)
    ]

    section_counts = [sum(1 for node in small_set if node in section) for section in sections]

    # Find the index of the dominant section
    dominant_section_index = section_counts.index(max(section_counts))

    # Return results
    return dominant_section_index, section_counts

def generate_pairs(list1, list2):
    return [(x, y) for x in list1 for y in list2]


def update_scores(scores_dict, small_set, big_set, combination_names):
    combinations = generate_pairs(small_set, big_set)
    for i, pair in enumerate(combinations):
        dominant_section, sections = check_nodes_position(pair[0], pair[1], no_of_parts)
        scores_dict[combination_names[i]] = (dominant_section, (sections))
    return scores_dict



all_users = open_pickle(os.path.join(base_dir, "selected_1_0_margin_politifact.pkl"))["user"]
print(f"No. of users: {len(all_users)}")

df_deg = pd.read_csv(os.path.join(base_dir, "network_analysis", 'user_degree_centrality.csv'))
top_k_users_deg_small = list(df_deg.nlargest(k_small_set, 'degree_centrality_score')['user_id'])
top_k_users_deg_big = list(df_deg.sort_values(by=['degree_centrality_score'], ascending=False)['user_id'])

df_pag = pd.read_csv(os.path.join(base_dir, "network_analysis", 'user_pagerank.csv'))
top_k_users_pag_small = list(df_pag.nlargest(k_small_set, 'pagerank_score')['user_id'])
top_k_users_pag_big = list(df_pag.sort_values(by=['pagerank_score'], ascending=False)['user_id'])

df_bet = pd.read_csv(os.path.join(base_dir, "network_analysis", 'user_betweenness_centrality.csv'))
top_k_users_bet_small = list(df_bet.nlargest(k_small_set, 'betweenness_centrality_score')['user_id'])
top_k_users_bet_big = list(df_bet.sort_values(by=['betweenness_centrality_score'], ascending=False)['user_id'])

df_clo = pd.read_csv(os.path.join(base_dir, "network_analysis", 'user_closeness_centrality.csv'))
top_k_users_clo_small = list(df_clo.nlargest(k_small_set, 'closeness_centrality_score')['user_id'])
top_k_users_clo_big = list(df_clo.sort_values(by=['closeness_centrality_score'], ascending=False)['user_id'])

node_ids_AL_margin = list(open_pickle(os.path.join(base_dir, "selected_1_0_margin_politifact.pkl"))["user"]) #top k: [:k], last k: [-k:]
node_ids_AL_entropy = list(open_pickle(os.path.join(base_dir, "selected_1_0_entropy_politifact.pkl"))["user"])
node_ids_lc = list(open_pickle(os.path.join(base_dir, "selected_1_0_leastconfidence_politifact.pkl"))["user"])

top_k_users_margin_big = node_ids_AL_margin #[:k_big_set]
#worst_k_users_margin_big = node_ids_AL_margin[-k_big_set:]
top_k_users_margin_small = node_ids_AL_margin[:k_small_set]

topk_users_entropy_big = node_ids_AL_entropy #[:k_big_set]
#worst_k_users_entropy_big = node_ids_AL_entropy[-k_big_set:]
topk_users_entropy_small = node_ids_AL_entropy[:k_small_set]

topk_users_lc_big = node_ids_lc #[:k_big_set]
#worst_k_users_lc = node_ids_lc[-k_big_set:]
topk_users_lc_small = node_ids_lc[:k_small_set]

centrality_small = [top_k_users_deg_small, top_k_users_bet_small, top_k_users_clo_small, top_k_users_pag_small]
centrality_big = [top_k_users_deg_big, top_k_users_bet_big,top_k_users_clo_big, top_k_users_pag_big]

AL_small = [top_k_users_margin_small, topk_users_entropy_small, topk_users_lc_small]
AL_big = [top_k_users_margin_big, topk_users_entropy_big, topk_users_lc_big]


#centrality in AL
combinations_centrality_in_AL_names = ["DC-margin", "DC-entropy", "DC-lc", "BC-margin", "BC-entropy", "BC-lc",
                      "CC-margin", "CC-entropy", "CC-lc", "PR-margin", "PR-entropy", "PR-lc"]
scores = update_scores(scores_dict={}, small_set=centrality_small, big_set=AL_big, combination_names=combinations_centrality_in_AL_names)


#AL in centrality
combinations_AL_in_centrality_names = ["margin-DC", "margin-BC", "margin-CC", "margin-PR",
                                       "entropy-DC", "entropy-BC", "entropy-CC", "entropy-PR",
                                       "lc-DC", "lc-BC", "lc-CC", "lc-PR"]
scores = update_scores(scores_dict=scores, small_set=AL_small, big_set=centrality_big, combination_names=combinations_AL_in_centrality_names)

#centrality in centrality
combination_centrality_names = ["DC-DC", "DC-BC", "DC-CC", "DC-PR", "BC-DC", "BC-BC", "BC-CC", "BC-PR",
                                "CC-DC", "CC-BC", "CC-CC", "CC-PR", "PR-DC", "PR-BC", "PR-CC", "PR-PR"]
scores = update_scores(scores_dict=scores, small_set=centrality_small, big_set=centrality_big, combination_names=combination_centrality_names)

#AL in AL
combinations_AL_names = ["margin_margin", "margin-entropy", "margin-lc",
                         "entropy-margin", "entropy-entropy", "entropy-lc",
                         "lc-margin", "lc-entropy", "lc-lc"]
scores = update_scores(scores_dict=scores, small_set=AL_small, big_set=AL_big, combination_names=combinations_AL_names)


save_dict_to_pickle(scores, os.path.join(base_dir, "network_analysis", f'user_centrality_AL_score_{k_small_set}.pkl'))
print(scores)

scores_summary = {k: scores[k][0] for k in scores}
print(scores_summary)


#scores = open_pickle(os.path.join(base_dir, "network_analysis", 'user_centrality_AL_score.pkl'))
#print(scores)