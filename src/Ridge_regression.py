

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


def Vec_Ridge(product, vectorizer=CountVectorizer()):


    ''' 

    Input product data frame
    Returns train/test split 
    Vectorizes data, and creates dictionary of {index:feature_name}
    Inputs vectorized data into a ridge regression to get coefficents of each features
    Returns top negative values and top positive values

    ''' 

    #Returns train/test split 
    X_train, X_test, y_train, y_test = train_test_split(product['key_words'], product['sentiment_score'], test_size=0.33, random_state=42)


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
    
    df = pd.DataFrame({'Feature_names':top_ten_list, 'Coeff': clf.coef_[top_ten]})

    df['Sentiment'] = df['Coeff'].apply(get_sentiment)
    
    return df
    

def get_sentiment(value):
    if value > 0: 
        return "Postive"
    else: 
        return "Negative"