import tweepy

api_key = ""
api_secret = ""

auth = tweepy.OAuthHandler(api_key, api_secret)
api = tweepy.API(auth)

def get_tweets(keyword):
    tweets = api.search_tweets(keyword, count=10)
    return [t.text for t in tweets]