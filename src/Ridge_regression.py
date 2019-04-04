

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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


    

def Vec_Ridge(Product_A, Product_B, Product_AB, vectorizer=CountVectorizer()):


    ''' 

    Input product_AB data frame
    Returns train/test split 
    Vectorizes data, and creates dictionary of {index:feature_name}
    Inputs vectorized data into a ridge regression to get coefficents of each features
    Returns top negative values and top positive values

    ''' 

    #Returns train/test split 
    X_train, X_test, y_train, y_test = train_test_split(Product_AB['key_words'], Product_AB['sentiment_score'], test_size=0.33, random_state=42)


    #Vectorizes data, and creates dictionary of {index:feature_name}

    training_features = vectorizer.fit_transform(X_train)
    test_features = vectorizer.transform(X_test)

    L = vectorizer.get_feature_names() 
    Dict_features = {idx: value for idx, value in enumerate(L)}


    #Inputs vectorized data into a ridge regression to get coefficents of each features
    clf = Ridge(alpha=1.0)
    clf.fit(training_features, y_train) 
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
        normalize=False, random_state=None, solver='auto', tol=0.001)

    y_pred = clf.predict(test_features)

    #Gets top ten features based on largest coefficents!!! (Here should return largest/ smallest for negative & positive)
    top_ten = np.argsort(clf.coef_)[:-10 - 1:-1]
    top_ten_list = []
    for i in top_ten:
        top_ten_list.append(Dict_features[i])
    
    
    return sent_dict(Product_A, Product_B, top_ten_list)
    







