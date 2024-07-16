import numpy as np
import datetime
import pandas as pd
from torch import cuda
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
device = 'cuda' if cuda.is_available() else 'cpu'

# load in tweets and stock prices for apple
aapl_tweets_df = pd.read_csv('labelled_tweets/aapl-full.csv',dtype={"datetime": str})
aapl_prices_df = pd.read_csv('stock_prices/AAPL_prices_2009-06-01_to_2020-08-01.csv')

# convert dates in the databases into datetime objects for easier manipulation
def convert_tweet_dates(df):
    dates = []
    for i,tweet in df.iterrows():
        date = tweet['datetime'][:10]
        tweet_date_obj = datetime.datetime.strptime(date, "%Y-%m-%d") # %H:%M:%S")
        dates.append(tweet_date_obj)
    return dates
def convert_prices_dates(df):
    dates = []
    for i,price in df.iterrows():
        date = price['Date'].split(' ')[0]
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        dates.append(date_obj)
    return dates
aapl_tweets_df.insert(0, 'date_obj', convert_tweet_dates(aapl_tweets_df))
aapl_prices_df.insert(0, 'date_obj', convert_prices_dates(aapl_prices_df))


# only select the columns we need
aapl_tweets_df = aapl_tweets_df[['date_obj','label']]
aapl_prices_df = aapl_prices_df[['date_obj','Open','High','Low','Close','Volume']]
# create the target to predict, whether the stock rises or falls the next day
aapl_prices_df['NextDayClose'] = aapl_prices_df['Close'].shift(-1)
aapl_prices_df['CloseChange'] = (aapl_prices_df['NextDayClose'] - aapl_prices_df['Close'])/aapl_prices_df['Close'] # * 100.0
aapl_prices_df['ChangeTrend'] = (aapl_prices_df['CloseChange'] > 0).astype(int)

#truncate data way outside the dates we have tweets for
# print(min(aapl_tweets_df['date_obj']))
# print(max(aapl_tweets_df['date_obj']))
aapl_prices_df = aapl_prices_df[(aapl_prices_df['date_obj'] >= min(aapl_tweets_df['date_obj'])) & (aapl_prices_df['date_obj'] <= max(aapl_tweets_df['date_obj']) )]  

# calculate an average sentiment score for all tweets on a particular day
sum_tweet_scores = {}
sum_tweet_score = 0
tweet_count = 1
date = aapl_tweets_df['date_obj'][0]
for i,tweet in aapl_tweets_df.iterrows():
    sentiment_label = 1
    if tweet['label'] == 0:
        sentiment_label = -1
    if tweet['label'] == 2:
        sentiment_label = 0

    if tweet['date_obj'] == date:
        sum_tweet_score += sentiment_label 
        tweet_count += 1
    else:
        date_obj = tweet['date_obj']
        sum_tweet_scores[date_obj] = sum_tweet_score/tweet_count
        sum_tweet_score = 0
        tweet_count = 1
        date = tweet['date_obj']
aapl_prices_df.insert(0, 'sum_tweet_scores', [0]*aapl_prices_df.shape[0])
for i, r in aapl_prices_df.iterrows():
    date_obj = r['date_obj']
    aapl_prices_df.at[i, 'sum_tweet_scores'] = sum_tweet_scores[date_obj]

features = ['Open','High','Low','Close','Volume','sum_tweet_scores','ChangeTrend']
features_df = aapl_prices_df[features]
# number of days for which historical data is available to the model
lookback_days = 10 # days
for i in range(1,lookback_days+1):
    features_df[f'close_prev_{i}'] = features_df['Close'].shift(i)
    features_df = features_df.dropna()
trend_df = features_df['ChangeTrend']
features_df = features_df.drop(['ChangeTrend'], axis=1)

# normalize the features
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(features_df)
scaled_features = scaler.fit_transform(features_df)
features_df = pd.DataFrame(scaled_features, columns=features_df.columns)
 
# drop the tweet scores column to see how the model performs with just financial data (ie RF1)
# features_df = features_df.drop(['sum_tweet_scores'], axis=1)

# number of test days, out of 500 total data points
split = 120

#create RF classifier
model = RandomForestClassifier(n_estimators=1000, min_samples_split=50, random_state=42)
model.fit(features_df[:-split], trend_df[:-split])

predictions = model.predict(features_df[-split:])
predictions = pd.Series(predictions, index=features_df[-split:].index)
print(f'f1 score: {f1_score(trend_df[-split:], predictions)}')
