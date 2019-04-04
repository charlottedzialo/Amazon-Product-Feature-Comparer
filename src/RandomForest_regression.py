
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor 
import numpy as np
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





def RF(Product_A, Product_B, Product_AB, vectorizer=CountVectorizer()):


    ''' 

    Input product data frame
    Returns train/test split 
    Vectorizes data, and creates dictionary of {index:feature_name}
    Inputs vectorized data into a ridge regression to get coefficents of each features
    Returns top negative values and top positive values

    ''' 

    #Returns train/test split 
    
    y = Product_AB['sentiment_score']


    #Vectorizes data, and creates dictionary of {index:feature_name}

    X = vectorizer.fit_transform(Product_AB['key_words'])
    

    L = vectorizer.get_feature_names() 
    Dict_features = {idx: value for idx, value in enumerate(L)}


    
    regr = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=100)
    regr.fit(X, y)
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=3,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
               oob_score=False, random_state=0, verbose=0, warm_start=False)
    

    #Gets top ten features based on largest coefficents!!! (Here should return largest/ smallest for negative & positive)
    top_ten = np.argsort(regr.feature_importances_)[:-10 - 1:-1]
    top_ten_list = []
    for i in top_ten:
        top_ten_list.append(Dict_features[i])
    
#     df = pd.DataFrame({'Feature_names':top_ten_list, 'Coeff': clf.coef_[top_ten]})

#     df['Sentiment'] = df['Coeff'].apply(get_sentiment)
    
    return sent_dict(Product_A, Product_B, top_ten_list)
    


