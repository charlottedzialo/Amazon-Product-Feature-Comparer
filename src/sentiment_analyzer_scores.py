from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    
    
    ''' 
    Returns Compound score for each sentence.
    The Compound score is a metric that calculates 
    the sum of all the lexicon ratings 
    which have been normalized between 
    -1(most extreme negative) and +1 (most extreme positive)
    
    '''
    score = analyser.polarity_scores(sentence)
    return score['compound']