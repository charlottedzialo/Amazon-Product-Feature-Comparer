
import pandas as pd

def sent_dict(prod_a, prod_b, lst):
    sent_dict_a = {}
    sent_dict_b = {}
    for word in lst:
        sent_dict_a[word] = [prod_a[prod_a['key_words'].str.contains(word)]['sentiment_score'].mean(),
                      prod_a[prod_a['key_words'].str.contains(word)]['key_words'].count()] 
        
        sent_dict_b[word] = [prod_b[prod_b['key_words'].str.contains(word)]['sentiment_score'].mean(),
                      prod_b[prod_b['key_words'].str.contains(word)]['key_words'].count()] 
        
    df_a = pd.DataFrame.from_dict(sent_dict_a, orient='index', 
                                  columns=['Product_A_avg_sentiment_score', 'Product_A_review_count'])
    
    df_b = pd.DataFrame.from_dict(sent_dict_b, orient='index',
                                  columns=['Product_B_avg_sentiment_score', 'Product_B_review_count'])
    
    
    result = pd.concat([df_a, df_b], axis=1)
                       
 
    return result
    