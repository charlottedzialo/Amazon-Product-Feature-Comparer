from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def tokenize(text):
    
    ''' 
    1. Tokenization: Split reviews into words
    2. Words that have fewer than 3 characters are removed.
    3. All stopwords are removed.
    4. Words are lemmatized — words in third person are changed to first person and
       verbs in past and future tenses are changed into present.
    5. Words are stemmed — words are reduced to their root form.
    
    '''
    

    tokens = text.split() 
    
    # removes stop words 
    STOPWORDS = set(stopwords.words('english'))
    no_stopwords = [w for w in tokens if not w in STOPWORDS]
    
    STEMMER = PorterStemmer()
    stemmed_lemmed = [STEMMER.stem(WordNetLemmatizer().lemmatize(w)) for w in no_stopwords]
    
    return [w for w in stemmed_lemmed if w]

