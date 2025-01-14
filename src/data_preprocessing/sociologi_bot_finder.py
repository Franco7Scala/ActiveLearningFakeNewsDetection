import pandas


path = "/home/scala/datasets/politifact/tweet_embedding_politifact.csv"
column_to_analyze = "source"

data_frame = pandas.read_csv(path)
unique_values = data_frame[column_to_analyze].unique()
print(f"Unique values in '{column_to_analyze}' column:")

for value in unique_values:
    print(value[value.rfind('">')+2: value.rfind('</a>')])
