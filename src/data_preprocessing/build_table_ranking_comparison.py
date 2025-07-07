from src.data_utils import open_pickle


base_dir = f"/home/scala/projects/Sociologi/data/ranking_comparison"

ranking_type = f"user"
dataset_name = f"mumin"

files_to_open = [f"{base_dir}/{dataset_name}/ranking_comparison_centrality_measures_{ranking_type}.pkl",
                 f"{base_dir}/{dataset_name}/ranking_comparison_post_hoc_techniques_{ranking_type}.pkl"]


for file_name in files_to_open:
    print(f"{dataset_name} {ranking_type} {file_name[file_name.find('ranking_comparison_') + 19: file_name.rfind('_')]}")
    dict = open_pickle(file_name)

    first_row = f"Couple\t"
    for i_key in dict[sorted(dict.keys())[0]].keys():
        first_row += f"{i_key}\t"

    print(first_row)

    for key in dict.keys():
        row = f"{key}\t"

        for i_key in dict[key].keys():
            row += f"{dict[key][i_key]}\t"

        print(row.replace(".", ","))

    print()

