from twikit import Client
import json
import pandas as pd

stock_name = '' #capital letters

client = Client('en-US')

# auth_info_1 = user name
# auth_info_2 = email

print('start login')
client.login(auth_info_1='Tassadar_Khalai',
             auth_info_2='danver.zhao@gmail.com',
             password='')

print('login success')
client.save_cookies('cookies.json')
client.load_cookies(path='cookies.json')


tweets_to_store = []
for y in range(1, 4):
    month = str(y).zfill(2)
    for i in range(1, 30):
        date = str(i).zfill(2)
        until_date = str(i+1).zfill(2)
        query = f'({stock_name}) lang:en until:2024-{month}-{until_date} since:2024-{month}-{date}'

        tweets = client.search_tweet(query=query, product="Top")

        for tweet in tweets:
            tweets_to_store.append({
                'created_at': tweet.created_at,
                'favorite_count': tweet.favorite_count,
                'retweet_count': tweet.retweet_count,
                'view_count': tweet.view_count,
                'text': tweet.text
            })
        
df = pd.DataFrame(tweets_to_store)
df.to_csv(f"evaluation_tweets_{stock_name}.csv")


