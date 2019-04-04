
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor 
import numpy as np





def RF(product, vectorizer=CountVectorizer()):


    ''' 

    Input product data frame
    Returns train/test split 
    Vectorizes data, and creates dictionary of {index:feature_name}
    Inputs vectorized data into a ridge regression to get coefficents of each features
    Returns top negative values and top positive values

    ''' 

    #Returns train/test split 
    
    y = product['sentiment_score']


    #Vectorizes data, and creates dictionary of {index:feature_name}

    X = vectorizer.fit_transform(product['key_words'])
    

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
    
    return top_ten_list