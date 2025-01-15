import pandas as pd

from src.data_utils import open_pickle, get_base_dir
import os

dir_base = get_base_dir()
fname = "user_label_stats.pkl"


def load_user_labels():
    return pd.read_pickle(os.path.join(dir_base, "user_labels.csv"))


def extract_ordered_users_discussions():
    user_stats_summary = open_pickle(os.path.join(dir_base, fname))
    sorted_users = sorted(user_stats_summary, key=lambda x: x[1][2], reverse=True)
    return sorted_users

def generate_user_label(x):
    num_true, num_false, num_total = int(x[0]), int(x[1]), int(x[2])
    difference = abs(num_true - num_false) / num_total if num_total != 0 else 0

    if difference < 0.2:  # Less than 20% difference
        return "mixed"
    elif num_true > num_false:  # More true values
        return "majority_true"
    else:  # More false values
        return "majority_false"


def assign_label_to_users_subset(user_stats_summary, user_ids):
    df = pd.DataFrame(columns=["user_id", "label"])
    df["user_id"] = [u[0] for u in user_stats_summary]
    df["label"] = [u[1] for u in user_stats_summary]
    df = df.loc[df["user_id"].isin(user_ids)]
    df["label"] = df["label"].apply(generate_user_label)
    return df


k=100
sorted_users = extract_ordered_users_discussions()
top_k_users = sorted_users[:k]

print(f"Top {k} users based on claims discussed:")
for user, (true_claims, false_claims, total_claims) in top_k_users:
    print(f"User {user}: True Claims = {true_claims}, False Claims = {false_claims}, Total Claims = {total_claims}")

df = assign_label_to_users_subset(sorted_users, list(range(len(sorted_users))))
df.to_csv(os.path.join(dir_base, "user_labels.csv"), index=False)
print(df.head())


