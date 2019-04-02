import unicodedata
import re
import numpy as np 


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
