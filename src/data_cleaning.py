import pandas as pd
import unicodedata
import re
import numpy as np 
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()






def text_cleaner(name): 
    
    
    ''' 
    Text cleaner is called in the clean data function. 
    Takes in text and cleans it. 
    
    '''
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', name)
    cleantext = str(cleantext).lower()
    cleaned = re.sub(r'[?|!|\'|"|#|$|%]',r'',cleantext)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = str(cleaned).lower()
    
    #removing accented characters
    cleaned = unicodedata.normalize('NFKD', cleaned).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    return cleaned








"""Create a list of common words to remove"""
stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
            "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
            "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", 
            "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", 
            "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", 
            "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", 
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", 
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]



nlp = en_core_web_sm.load()


def get_aspects(x):

    
    """Apply the function to get aspects from reviews"""

    doc=nlp(x) ## Tokenize and extract grammatical components
    doc=[i.text for i in doc if i.text not in stop_words and i.pos_=="NOUN"] ## Remove common words and retain only nouns
    doc=list(map(lambda i: i.lower(),doc)) ## Normalize text to lower case
    doc=pd.Series(doc)
    doc=doc.value_counts().head(11).index.tolist() ## Get 5 most frequent nouns
    return doc



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


def data_cleaning(url):
   
   ''' 
   1. create new columns with review headline + review text
   2. create new df  
   3. Remove products with no reviews
   4. Calls clean text, to clean the text reviews
   5. Drops old text review
   6. Calculates the sentiment 
   7. Calls get_aspects to get key words
   
   
   '''
   cols=['marketplace', 'customer_id', 'review_id', 'product_id',
      'product_parent', 'product_title', 'product_category', 'star_rating',
      'helpful_votes', 'total_votes', 'vine', 'verified_purchase',
      'review_headline', 'review_body', 'review_date'] 


   ## GOING TO HAVE TO CHANGE THIS SO GETS THE TWO PRODUCT NAMES 
   df = pd.read_csv(url, sep = '\t', names = cols)
   df['review_text'] = df['review_headline']+". "+ df['review_body']
   df = pd.concat([df['product_id'], 
                  df['product_title'], 
                  df['review_text'],
                  df['star_rating']],axis=1)
   
   
   df = df[pd.notnull(df['review_text'])]
   
   df['clean_text'] = df['review_text'].apply(text_cleaner)
   
   df = df.drop('review_text', axis=1)
   
   df['sentiment_score'] = df['clean_text'].apply(sentiment_analyzer_scores)
   
   df['key_words'] = df['clean_text'].apply(get_aspects)
   
   df['key_words'] = df['key_words'].apply(', '.join)

   
   return df 