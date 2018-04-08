# Sentiment_Analysis

This contains Python Scripts for Sentiment Analysis of Movie Review . The dataset has been taken from Stanford's website (http://ai.stanford.edu/~amaas/data/sentiment/) . It basically contains around 50,000 movie reviews from the Imdb Website . Positive reviews are in pos folder and negative reviews are in neg folder. This script uses different features along with different classification algorithms . 
#The features used are ::

### Binary bag of words.
### Normalized Term frequency (tf) representation.
### Tfidf representation.
### Average of the Word2vec word vectors in the document with and without tfidf weights for each word vector while averaging ( pretrained Google Word2Vec representation ).
Download the Google pretrained Word2Vec file before executing .
### GLoVE vector representations for words ( pretrained Stanford Glove representation ).
Download Stanford pretrained Glove reoresentation.

# Classifications used ::

### Logistic Regression
### stochastic gradient descent
### Naive Bayes
