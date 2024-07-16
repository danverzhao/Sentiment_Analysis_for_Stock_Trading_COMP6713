from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    # Create a SentimentIntensityAnalyzer object
    analyzer = SentimentIntensityAnalyzer()

    # Analyze the sentiment of the text
    scores = analyzer.polarity_scores(text)

    # Extract the sentiment scores
    negative = scores['neg']
    neutral = scores['neu']
    positive = scores['pos']
    compound = scores['compound']

    # Determine the overall sentiment based on the compound score
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment, scores

# Example usage
text = "apple stock is going to fall today"
sentiment, scores = analyze_sentiment(text)
print(f"Sentiment: {sentiment}")
print(f"Sentiment Scores: {scores}")