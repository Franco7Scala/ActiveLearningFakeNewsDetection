import pickle

import numpy as np
import pandas as pd
import os
import torch
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.data_preprocessing.politifact import load_politifact_heterodata
from src.data_utils import get_base_dir, open_pickle
from src.network_analysis.important_nodes_analysis_v2 import construct_user_graph
from src.network_analysis.topk_users_stats import base_dir
from src.network_analysis.user_labels import extract_ordered_users_discussions


# GENERAL PARAMETERS
embeddings_dir = "/mnt/nas/scala/politifact"
topk = 50

# UMAP
n_neighbors = 5
min_dist = 0.1
metric = 'cosine'

#COLORS
fake_color = '#FF0000'  # Red for 'fake'
real_color = '#89CFF0'  # Light blue for 'real'
mixed_color = '#808080' # grey for 'mixed'
default_color = '#D3D3D3' # light grey if label not found

color_map_news = {'real': real_color, 'fake': fake_color}
color_map_user = {'real': real_color, 'fake': fake_color, 'mixed': mixed_color}


def int_to_label(x):
    return 'real' if x == 0 else 'fake'


def load_user_label_df():
    df = pd.read_csv(os.path.join(get_base_dir(), "user_labels.csv"))
    df["label"] = df["label"].map({'majority_true': "real", 'majority_false': "fake", 'mixed': 'mixed'})
    return df


def load_classes_politifact(node_type):
    if node_type == 'user':
        df = load_user_label_df()
        return df['label'].values
    else:
        #df = pd.read_csv(os.path.join(get_base_dir(), "original_data", "nodes", "news.csv"))
        #df["label"] = df["label"].map({'0': "real", '1': "fake"})
        Y = torch.load(os.path.join(get_base_dir(), "heterodata", "NY_tensor.pt"))
        Y_np = Y.numpy()
        return np.vectorize(int_to_label)(Y_np)


def load_embeddings(node_type, embeddings_dir):
    embs = open_pickle(os.path.join(embeddings_dir, "embeddings_after_training.pkl"))
    return embs[node_type].detach().cpu().numpy()


def extract_embeddings_and_labels_subset(embeddings, labels, ids_to_keep):
    return embeddings[ids_to_keep], labels[ids_to_keep]


def extract_users_by_label(df, label):
    return df[df['label'] == label]['user_id'].tolist()


#topk utenti per numero totale di news discusse per ogni label
def extract_topk_user_ids_by_label_politifact(k, label):

    df_user_label = load_user_label_df()
    users_labeled = extract_users_by_label(df=df_user_label, label=label)
    sorted_users = extract_ordered_users_discussions()
    sorted_users_labeled = [u for u in sorted_users if u[0] in users_labeled]
    topk_users = [user for user, _ in sorted_users_labeled[:k]]

    #heterodata = load_politifact_heterodata()
    #_, user_id_map = construct_user_graph(heterodata) #id_str: id_int

    #topk_users_ids = [user_id_map[id_str] for id_str in topk_users] #TODO user_id_maps sono in numero e tipo giusto ma valore sballato
    topk_users_ids = topk_users
    return topk_users_ids


def main():
    # load embeddings and labels
    print("loading embeddings...")
    embeddings_news = load_embeddings(node_type="news", embeddings_dir=embeddings_dir)
    labels_news = load_classes_politifact(node_type="news")

    embedding_user = load_embeddings(node_type="user", embeddings_dir=embeddings_dir)
    labels_user = load_classes_politifact(node_type="user")

    '''
    print("computing topk nodes and generating subset...")
    # topk users aka indices to keep. Non interessa più l'ordinamento decrescente
    topk_users_real = extract_topk_user_ids_by_label_politifact(k=topk, label="real")
    topk_users_fake = extract_topk_user_ids_by_label_politifact(k=topk, label="fake")
    topk_users_mixed = extract_topk_user_ids_by_label_politifact(k=topk, label="mixed")
    topk_users = sorted(topk_users_real+topk_users_fake+topk_users_mixed)

    with open(os.path.join(base_dir, 'topk_users_labels.pkl'), 'wb') as file:
        pickle.dump(topk_users, file)
    '''
    topk_users = open_pickle(os.path.join(base_dir, 'topk_users_labels.pkl'))

    print(max(topk_users), min(topk_users))
    embedding_user, labels_user = extract_embeddings_and_labels_subset(embeddings=embedding_user,
                                                                       labels=labels_user,
                                                                       ids_to_keep=topk_users)

    print("umap & plot...")
    #umap
    umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)

    umap_embeddings_news = umap_model.fit_transform(embeddings_news)
    umap_embeddings_user = umap_model.fit_transform(embedding_user)

    colors_news = np.array([color_map_news.get(label, default_color) for label in labels_news])  # Default to light gray if label not found
    colors_user = np.array([color_map_user.get(label, default_color) for label in labels_user])

    # Plotting
    plt.figure(figsize=(12, 10))

    # Plot news embeddings (triangles)
    plt.scatter(
        umap_embeddings_news[:, 0], umap_embeddings_news[:, 1],
        c=colors_news,
        marker='^',  # Triangle marker
        s=100,  # Size of the marker
        edgecolor='k',  # Black edge for visibility
        alpha=0.7
    )

    # Plot user embeddings (circles)
    plt.scatter(
        umap_embeddings_user[:, 0], umap_embeddings_user[:, 1],
        c=colors_user,
        marker='o',  # Circle marker
        s=50,  # Size of the marker
        edgecolor='k',  # Black edge for visibility
        alpha=0.7
    )

    # Create a custom legend
    handles = [
        patches.Patch(color=real_color, label='Real (News & User)'),
        patches.Patch(color=fake_color, label='Fake (News & User)'),
        patches.Patch(color=mixed_color, label='Mixed (User Only)')
    ]
    plt.legend(handles=handles, loc='upper right', fontsize=12, title="Legend", title_fontsize=14)

    # Save the plot as a PDF file
    plt.savefig(os.path.join(get_base_dir(), 'umap_embeddings.pdf'), format='pdf', bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main()

