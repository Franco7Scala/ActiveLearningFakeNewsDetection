import pandas

output_path = "/home/scala/datasets/politifact/user2bot.csv"

path_tweets_bot = "/home/scala/datasets/politifact/tweet_embedding_politifact_with_bot.csv"
path_tweets_discusses_news = "/home/scala/datasets/politifact/tweet_discusses_news.csv"
#path_news = "/home/scala/datasets/politifact/news_politifact.csv"
path_user_post_tweets = "/home/scala/datasets/politifact/user_posted_tweet.csv"

df_tweets_bot = pandas.read_csv(path_tweets_bot)
df_tweets_discusses_news = pandas.read_csv(path_tweets_discusses_news)
#df_news = pandas.read_csv(path_news)
df_user_post_tweets = pandas.read_csv(path_user_post_tweets)

df_user_post_tweets = df_user_post_tweets.rename(columns={"src": "user_that_writes"})
#df_news = df_news[["news_id"]]

join_1 = df_tweets_bot.join(df_tweets_discusses_news, lsuffix="tweet_id", rsuffix="src", how="left")     # join tweet_bot with tweet_news
join_1 = join_1.rename(columns={"tgt": "id_news"})
join_2 = join_1.join(df_user_post_tweets, lsuffix="tweet_id", rsuffix="tgt", how="left")                 # join previous with user_that_posted_tweet

selected = join_2[["user_that_writes", "bot_generated"]]

dict_users_true = {}
dict_users_false = {}

for index, row in selected.iterrows():
    dict_users_true[row["user_that_writes"]] = 0
    dict_users_false[row["user_that_writes"]] = 0

for index, row in selected.iterrows():
    if row["bot_generated"]:
        if row["user_that_writes"] in dict_users_true:
            dict_users_true[row["user_that_writes"]] += 1

    else:
        if row["user_that_writes"] in dict_users_false:
            dict_users_false[row["user_that_writes"]] += 1

responses = []
for key in dict_users_true.keys():
    responses.append((key, dict_users_true[key] > dict_users_false[key]))

result = pandas.DataFrame(responses, columns=("user_that_writes", "bot_generated"))
result.to_csv(output_path, index=False)
