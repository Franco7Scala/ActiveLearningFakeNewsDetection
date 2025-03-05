import os
from src.data_utils import open_pickle, get_base_dir

my_list = open_pickle(os.path.join(get_base_dir(), 'user_stats.pkl'))
count = sum(1 for _, inner_tuple in my_list if inner_tuple[2] > 10)

print(f"Number of external tuples with the third element greater than 10: {count}")