n_samples = 10000
n_features = 10000
n_components = 10
n_top_words = 10


def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)




def NMF(text):
    ''' Use tf-idf features for NMF ''' 

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')

    tfidf = tfidf_vectorizer.fit_transform(prod_B["clean_text"])




    # Fit the NMF model with tf-idf features

    nmf = NMF(n_components=n_components, random_state=1,
            alpha=.1, l1_ratio=.5).fit(tfidf)


    # Topics in NMF model 

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)



def LDA(text)

'''' fitting LDA models with tf features'''' 

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')

    tf = tf_vectorizer.fit_transform(prod_B["clean_text"])



    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)