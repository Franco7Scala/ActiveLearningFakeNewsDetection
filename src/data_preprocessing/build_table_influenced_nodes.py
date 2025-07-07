from src.data_utils import open_pickle


base_dir = f"/home/scala/projects/Sociologi/data/ranking_comparison"

ranking_type = f"user"
dataset_name = f"mumin"

files_to_open = [f"{base_dir}/{dataset_name}/influenced_nodes_{ranking_type}_count_all.pkl",
                 f"{base_dir}/{dataset_name}/influenced_nodes_{ranking_type}_count_checkpoint.pkl",
                 f"{base_dir}/{dataset_name}/influenced_nodes_from_combinations_{ranking_type}_count_all.pkl",
                 f"{base_dir}/{dataset_name}/influenced_nodes_from_combinations_{ranking_type}_count_checkpoint.pkl"]

post_hoc_techniques_names = ["GNN_expl_feats", "GNN_expl_rel", "AL_entropy", "AL_lcs", "AL_margin"]
centrality_measures_names = ["degree", "pagerank", "voterank", "betweenness", "closeness"]
single_names = post_hoc_techniques_names + centrality_measures_names
pairs_names = ["GNN_expl_feats - AL_entropy", "GNN_expl_feats - AL_lcs", "GNN_expl_feats - AL_margin", "GNN_expl_rel - AL_entropy", "GNN_expl_rel - AL_lcs", "GNN_expl_rel - AL_margin"]

if dataset_name == "politifact":
    if ranking_type == "news":
        div_factor = 696

    else:
        div_factor = 169106

elif dataset_name == "mumin":
    if ranking_type == "claim":
        div_factor = 2168

    else:
        div_factor = 153168

else:
    raise ValueError(f"Unknown dataset: {dataset_name}")

for file_name in files_to_open:
    if "from_combinations" in file_name:
        names = pairs_names

    else:
        names = single_names

    dict = open_pickle(file_name)
    columns_2 = []
    columns_3 = []
    columns_2.append(["DEPT 2"] + names)
    columns_3.append(["DEPT 3"] + names)

    for key in dict.keys():
        column_2 = [round(div_factor / key)]
        column_3 = [round(div_factor / key)]
        for name in names:
            if name in dict[key][2].keys():
                column_2.append(dict[key][2][name])

            else:
                column_2.append("0")

        for name in names:
            if name in dict[key][3].keys():
                column_3.append(dict[key][3][name])

            else:
                column_3.append("0")

        columns_2.append(column_2)
        columns_3.append(column_3)

    columns_2 = list(map(list, zip(*columns_2)))
    columns_3 = list(map(list, zip(*columns_3)))

    #print("-" * 50 + file_name + "-" * 50)
    #print("+" * 10 + "2" + "+" * 10)
    print()
    print(f"{dataset_name} - {ranking_type} - {file_name[file_name.rfind('/') + 1:-4].replace('_', ' ')}")
    for i in range(len(columns_2)):
        line = ""
        for j in range(len(columns_2[0])):
            line += f"{columns_2[i][j]}\t"

        print(line)

    #print("+" * 10 + "3" + "+" * 10)
    print()
    print(f"{dataset_name} - {ranking_type} - {file_name[file_name.rfind('/') + 1:-4].replace('_', ' ')}")
    for i in range(len(columns_3)):
        line = ""
        for j in range(len(columns_3[0])):
            line += f"{columns_3[i][j]}\t"

        print(line)
