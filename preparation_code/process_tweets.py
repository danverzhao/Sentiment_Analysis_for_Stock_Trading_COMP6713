import re
import emoji
# remove hashtag symbols, replace urls with [URL], remove repeated punctuation (ie !!! --> !), replace emoji with text version ie ' :warning: '
def preprocess_tweet(tweet):
    tweet = tweet.replace('#', '')
    tweet = tweet.replace('\n', '')
    tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '[URL]', tweet)
    repeated_pattern = re.compile(r'([^\w\s])\1+')
    tweet = repeated_pattern.sub(r'\1', tweet)
    tweet = emoji.demojize(tweet, delimiters = (" :", ": "))
    repeated_pattern = re.compile(r':+')
    tweet = repeated_pattern.sub(':', tweet)
    return tweet    

# usage
# import pandas as pd
# train_tweets_aapl_df = pd.read_csv('evaluation_tweets_aapl.csv')
# train_tweets_tsla_df = pd.read_csv('evaluation_tweets_tsla.csv')

# train_tweets_aapl_df['text'] = train_tweets_aapl_df['text'].apply(preprocess_tweet)
# train_tweets_tsla_df['text'] = train_tweets_tsla_df['text'].apply(preprocess_tweet)


# train_tweets_aapl_df.drop(train_tweets_aapl_df.columns[0], axis=1, inplace=True)
# train_tweets_tsla_df.drop(train_tweets_tsla_df.columns[0], axis=1, inplace=True)
# train_tweets_aapl_df.to_csv('preprocessed-evaluation-tweets/aapl.csv', index=False)
# train_tweets_tsla_df.to_csv('preprocessed-evaluation-tweets/tsla.csv', index=False)

#testing
# text = '#test!??! ** water https://www.ne.com'
# print(preprocess_tweet(text))
