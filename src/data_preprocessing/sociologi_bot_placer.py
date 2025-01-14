import pandas


path = "/home/scala/datasets/politifact/tweet_embedding_politifact.csv"
path_bot = "/home/scala/datasets/politifact/tweet_bots.txt"
path_to_save = "/home/scala/datasets/politifact/tweet_embedding_politifact_with_bot.csv"
column_to_analyze = "source"


list_bots = []
with open(path_bot, "r") as file:
    for line in file:
        list_bots.append(line)


def check_value(value):
    for bot in list_bots:
        if bot.replace("\n", "") in value:
            return True

    return False


data_frame = pandas.read_csv(path)
result = []
for value in data_frame[column_to_analyze]:
    result.append(check_value(value))

data_frame["bot_generated"] = result
data_frame.to_csv(path_to_save)
