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
    