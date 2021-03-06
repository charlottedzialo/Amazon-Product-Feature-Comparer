# Tell us how you really feel.

In today's competitive landscape, organizations are expected to deliver products that exceed customer expectations. Knowing what customers are talking about and understanding their problems and opinions is the foundation of many successful products.

Product reviews hold valuable insights into customer sentiment. These insights can be used for competitive analysis, product design, and user experience. However, with a large volume of reviews and a variety of channels, it is getting harder and harder to process customer feedback. To stay competitive, organizations need a quick and easy way to gauge what features customers like and dislike about a product.

The "Tell us how you really feel" application enables users to easily see the top positive and negative features of a product derived directly from customer reviews. I used the iconic [Amazon Product Review Dataset](https://s3.amazonaws.com/amazon-reviews-pds/readme.html) for this project. This dataset is collection ***130+ million customer reviews*** written in the Amazon.com marketplace and associated metadata from 1995 until 2015.

# Modeling and Techniques

Approach 1: Predicted review sentiment to determine key features with Random Forest and Ridge regression.

Approach 2: Leveraged advanced NLP techniques and sentiment analysis to automatically extract and catagorize key noun phrases.

![Image of Workflow](https://github.com/charlottedzialo/Tell-us-how-you-really-feel/blob/master/Screen%20Shot%202019-05-13%20at%205.03.38%20PM.png)

# Results 

Advanced NLP with Spark provided the most relevant results compared to Random Forest and Ridge Regression models.

***Random Forest*** ----> tinny, attempts, connector, problem, stars, connection, speaker, waste, pot

***Ridge Regression*** ----> connectivity, one, length, speaker, ipod, design, ipad, gift, item, speakers

***Advanced NLP*** ----> great sound, good sound, hard understand, sound bad small device, happy purchase, speaker box                                  trouble, new pressure point volume, awesome, worth money, good range 
