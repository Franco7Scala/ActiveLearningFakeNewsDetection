from src.data_utils import open_pickle, get_base_dir

n_split = 1
cycle = 0

ids = open_pickle(f"{get_base_dir()}/mumin/selected_{n_split}_{cycle}.pkl")

print(ids)
