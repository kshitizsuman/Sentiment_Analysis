import nltk,gensim
from nltk.tokenize import word_tokenize, RegexpTokenizer
train_path = "aclImdb/train/" # source data
test_path = "./imdb_te.csv" # test data for grade evaluation. 
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from gensim.scripts.glove2word2vec import glove2word2vec
sw = set(stopwords.words('english'))

def glove_word2vec():
	glove_input_file = 'glove.6B.300d.txt'
	word2vec_output_file = 'glove2word2vec.txt'
	glove2word2vec(glove_input_file, word2vec_output_file)

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
	import pandas as pd 
	from pandas import DataFrame, read_csv
	import os
	import csv 
	import numpy as np 

	stopwords = open("stopwords.en.txt", 'r' , encoding="ISO-8859-1").read()
	stopwords = stopwords.split("\n")

	indices = []
	text = []
	rating = []

	i =  0 

	for filename in os.listdir(inpath+"pos"):
		data = open(inpath+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
		data = remove_stopwords(data, stopwords)
		indices.append(i)
		text.append(data)
		rating.append("1")
		i = i + 1

	for filename in os.listdir(inpath+"neg"):
		data = open(inpath+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
		data = remove_stopwords(data, stopwords)
		indices.append(i)
		text.append(data)
		rating.append("0")
		i = i + 1

	Dataset = list(zip(indices,text,rating))
	
	if mix:
		np.random.shuffle(Dataset)

	df = pd.DataFrame(data = Dataset, columns=['row_Number', 'text', 'polarity'])
	df.to_csv(outpath+name, index=False, header=True)

	pass

def tokenize(text):
	import nltk.stem
	tokens = nltk.word_tokenize(text)
	stems = []
	for item in tokens:
		stems.append(nltk.stem.PorterStemmer().stem(item))
	return stems


def remove_stopwords(sentence, stopwords):
	sentencewords = sentence.split()
	resultwords  = [word for word in sentencewords if word.lower() not in stopwords]
	result = ' '.join(resultwords)
	return result


def bag_of_words(data):
	from sklearn.feature_extraction.text import CountVectorizer
	vectorizer = CountVectorizer(stop_words='english')
	vectorizer = vectorizer.fit(data)
	return vectorizer

def tf_process(data):
	from sklearn.feature_extraction.text import TfidfVectorizer
	transformer = TfidfVectorizer(stop_words='english',use_idf=False,norm='l1')
	transformer = transformer.fit(data)
	return transformer


def tfidf_process(data):
	from sklearn.feature_extraction.text import TfidfVectorizer 
	transformer = TfidfVectorizer(stop_words='english')
	transformer = transformer.fit(data)
	return transformer



def retrieve_data(name="imdb_tr.csv", train=True):
	import pandas as pd 
	data = pd.read_csv(name,header=0, encoding = 'ISO-8859-1')
	X = data['text']
	
	if train:
		Y = data['polarity']
		return X, Y

	return X		

'''
def stochastic_descent(Xtrain, Ytrain, Xtest):
	from sklearn.linear_model import SGDClassifier 
	clf = SGDClassifier(loss="hinge", penalty="l1", n_iter=20)
	clf.fit(Xtrain, Ytrain)
	Ytest = clf.predict(Xtest)
	return Ytest
'''

def naive_bayes_classifierG(Xtrain,Ytrain,Xtest):
	from sklearn.naive_bayes import MultinomialNB,GaussianNB
	clf = GaussianNB().fit(Xtrain, Ytrain)
	Ytest = clf.predict(Xtest)
	return Ytest

def naive_bayes_classifier(Xtrain,Ytrain,Xtest):
	from sklearn.naive_bayes import MultinomialNB,GaussianNB
	clf = MultinomialNB().fit(Xtrain, Ytrain)
	Ytest = clf.predict(Xtest)
	return Ytest

def svm_classifier(Xtrain,Ytrain,Xtest):
	import sklearn.svm
	clf = sklearn.svm.SVC()
	clf.fit(Xtrain,Ytrain)
	Ytest = clf.predict(Xtest)
	return Ytest

def logistic_regression(Xtrain,Ytrain,Xtest):
	import sklearn.linear_model
	clf = sklearn.linear_model.LogisticRegression()
	clf.fit(Xtrain,Ytrain)
	Ytest = clf.predict(Xtest)
	return Ytest

def word_to_vec(data):
	model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True)
	print("PreTrained Representation has been loaded for Word2Vec!!")
	tokenizer = RegexpTokenizer(r'\w+')
	word2vec_rep = []
	for sentence in data:
		tokens = tokenizer.tokenize(sentence)
		varist = [word for word in tokens if not word in sw]
		
		temp = sum(model[word.lower()] for word in varist if word.lower() in model)
		count = sum(1 for word in varist if word.lower() in model)
		
		temp = temp/count
		k = word2vec_rep + [temp]
		
		word2vec_rep = k
	word2vec_rep = np.array(word2vec_rep)
	del model
	return word2vec_rep

def tfidfw2v(data):
	from sklearn.feature_extraction.text import TfidfVectorizer
	vectorizer = TfidfVectorizer(stop_words = sw)
	tfidf_vect = vectorizer.fit_transform(data)
	model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True)
	print("PreTrained Representation has been loaded for Tfidf Word2Vec!!")
	i=0
	tokenizer = RegexpTokenizer(r'\w+')
	tfidf_word2vec_rep = []
	for sentence in data:
		tokens = tokenizer.tokenize(sentence)
		i=i+1
		varist = [word for word in tokens if not word in sw]
		group = Counter(varist)
		grp = [key for key in group if key.lower() in model]
		
		x = tfidf_vect[i-1].toarray()
		
		temp = sum(model[key.lower()]*x[0][vectorizer.vocabulary_[key.lower()]] for key in grp if key.lower() in vectorizer.vocabulary_ )
		count = sum(group[key] for key in grp if key.lower() in vectorizer.vocabulary_ )
		
		k = tfidf_word2vec_rep + [temp/count]
		tfidf_word2vec_rep = k
	tfidf_word2vec_rep = np.array(tfidf_word2vec_rep)
	del model
	return tfidf_word2vec_rep
'''
def google_w2v(data):
	from sklearn import feature_extraction as fe
	from sklearn.metrics import accuracy_score as ac
	import numpy as np
	import pandas as pd
	from sklearn.naive_bayes import BernoulliNB
	from nltk.tokenize import wordpunct_tokenize, sent_tokenize
	from nltk.tokenize import word_tokenize
	from nltk.corpus import stopwords
	from string import punctuation
	from nltk.stem.wordnet import WordNetLemmatizer
	from nltk.corpus import wordnet
	import nltk
	import os.path
	import sklearn.linear_model
	import pickle
	import gensim,string
	i=0
	X_test=[]
	stop_words = set(stopwords.words('english')+list(punctuation))
	for rev in data:
		#temp=rev
		#tokens=word_tokenize(str(temp))
		#tokens=[w for w in tokens if not w in stop_words]
		#X_test.append(tokens)
		tokens=str(rev).split()
		table = str.maketrans('', '', string.punctuation)
		tokens = [w.translate(table) for w in tokens]
		tokens = [word for word in tokens if word.isalpha()]
		stop_words = set(stopwords.words('english'))
		tokens = [w for w in tokens if not w in stop_words]
		tokens = [word for word in tokens if len(word) > 1]
		X_test.append(tokens)
	model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary = True)
	print("Done !!")
	test_repr=[]
	for sentence in X_test:
		sumtok=[0]*300
		count =0
		for word in sentence:
			count =count+1
			if word in [model.wv.vocab]:
				w2vec = model[word]
				for i in range(100):
					sumtok[i] = sumtok[i]+w2vec[i]
		sumtok[:] = [x/count for x in sumtok]
		test_repr.append(sumtok)
	del model
	return test_repr

def w2v(data,ytrain,xtest,yori):
	from sklearn import feature_extraction as fe
	from sklearn.metrics import accuracy_score as ac
	import numpy as np
	import pandas as pd
	from sklearn.naive_bayes import BernoulliNB
	from nltk.tokenize import wordpunct_tokenize, sent_tokenize
	from nltk.tokenize import word_tokenize
	from nltk.corpus import stopwords
	from string import punctuation
	from nltk.stem.wordnet import WordNetLemmatizer
	from nltk.corpus import wordnet
	import nltk
	import os.path
	import sklearn.linear_model
	import pickle
	import gensim
	i=0
	X_train=[]
	stop_words = stop_words = set(stopwords.words('english')+list(punctuation))
	
	for j in range():
		temp=data[j]
		tokens=wordpunct_tokenize(str(temp))
		tokens=[w for w in tokens if not w in stop_words]
		X_train.append(tokens)
	print("Train file input")
	model = gensim.models.Word2Vec(X_train, size  = 100, window =5, min_count =2, workers =4)
	model.save('model.bin')

'''

def glove_word2vec():
	glove_input_file = 'glove.6B.100d.txt'
	word2vec_output_file = 'glove2word2vec.txt'
	glove2word2vec(glove_input_file, word2vec_output_file)

def glove_vec(data):
	#l_word2vec(corpus,gensim.models.KeyedVectors.load_word2vec_format("glove2word2vec.txt"),"glove_rep.txt")
	model = gensim.models.KeyedVectors.load_word2vec_format("glove2word2vec.txt")
	print("PreTrained Glove has been loaded !!")
	tokenizer = RegexpTokenizer(r'\w+')
	word2vec_rep = []
	for sentence in data:
		tokens = tokenizer.tokenize(sentence)
		varist = [word for word in tokens if not word in sw]
		temp = sum(model[word.lower()] for word in varist if word.lower() in model)
		tot = sum(1 for word in varist if word.lower() in model)
		temp = temp/tot
		k = word2vec_rep + [temp]
		word2vec_rep = k
	word2vec_rep = np.array(word2vec_rep)
	del model
	return word2vec_rep

def tfidf_glove_vec(data):
	#tfidf_word2vec_rep(corpus,gensim.models.KeyedVectors.load_word2vec_format("glove2word2vec.txt"),"tfidf_glove_rep.txt")
	from sklearn.feature_extraction.text import TfidfVectorizer
	vectorizer = TfidfVectorizer(stop_words = sw)
	tfidf_file = vectorizer.fit_transform(data)
	model = gensim.models.KeyedVectors.load_word2vec_format("glove2word2vec.txt")
	print("PreTrained Glove for Tfidf Glove has been loaded !!")
	i = 0
	tokenizer = RegexpTokenizer(r'\w+')
	tfidf_word2vec_rep = []
	for st in data:
		i = i+1
		tokens = tokenizer.tokenize(st)
		ist = [word for word in tokens if not word in sw]
		group = Counter(ist)
		grp = [key for key in group if key.lower() in model]
		x = tfidf_file[i-1].toarray()
		temp = sum(model[key.lower()]*x[0][vectorizer.vocabulary_[key.lower()]] for key in grp if key.lower() in vectorizer.vocabulary_ )
		tot = sum(group[key] for key in grp if key.lower() in vectorizer.vocabulary_ )
		
		k = tfidf_word2vec_rep + [temp/tot]
		tfidf_word2vec_rep = k
	tfidf_word2vec_rep = np.array(tfidf_word2vec_rep)
	del model
	return tfidf_word2vec_rep


def wordtovec(data):
	from sklearn import feature_extraction as fe
	from sklearn.metrics import accuracy_score as ac
	import numpy as np
	import pandas as pd
	from sklearn.naive_bayes import BernoulliNB
	from nltk.tokenize import wordpunct_tokenize, sent_tokenize
	from nltk.tokenize import word_tokenize
	from nltk.corpus import stopwords
	from string import punctuation
	from nltk.stem.wordnet import WordNetLemmatizer
	from nltk.corpus import wordnet
	import nltk
	import os.path
	import sklearn.linear_model
	import pickle
	import gensim
	i=0
	X_train=[]
	stop_words = stop_words = set(stopwords.words('english')+list(punctuation))
	for rev in data:
		temp=rev
		tokens=wordpunct_tokenize(str(temp))
		tokens=[w for w in tokens if not w in stop_words]
		X_train.append(tokens)

	print("Train file input")
	model = gensim.models.Word2Vec(X_train, size  = 100, window =5, min_count =2, workers =4)
	model.save('model.bin')
	#k = 


def neural_net(X_train,y_train,X_test,Y_test):
	from keras.preprocessing import sequence
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Activation
	from keras.layers import Embedding
	from keras.layers import Convolution1D, GlobalMaxPooling1D
	model = Sequential()
	model.add(Embedding(max_features, embedding_dims, input_length=maxlen, dropout=0.2))
	model.add(Convolution1D(nb_filter=nb_filter,filter_length=filter_length,border_mode='valid',activation='relu',subsample_length=1,))
	model.add(GlobalMaxPooling1D())
	model.add(Dense(hidden_dims))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	h = model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,validation_data=(X_test, y_test),verbose=1,)

'''
def accuracy(Ytrain, Ytest):
	assert (len(Ytrain)==len(Ytest))
	num =  sum([1 for i, word in enumerate(Ytrain) if Ytest[i]==word])
	n = len(Ytrain)  
	return (num*100)/n
'''
def accuracy(Ytest_polarity,Ytest):
	import sklearn.metrics
	return sklearn.metrics.accuracy_score(Ytest_polarity,Ytest) * 100
	#assert (len(Ytrain)==len(Ytest))
	#num =  sum([1 for i, word in enumerate(Ytest_polarity) if Ytest[i]==word])
	#n = len(Ytest_polarity)  
	#return (num*100)/n


def write_txt(data, name):
	data = ''.join(str(word) for word in data)
	file = open(name, 'w')
	file.write(data)
	file.close()
	pass 


if __name__ == "__main__":
	import time
	start = time.time()
	print ("Processing the training_data --")
	imdb_data_preprocess(inpath=train_path, mix=True)
	print ("CSV Generated.")
	[Xtrain_text, Ytrain] = retrieve_data()
	print ("Data Retrieved")
	imdb_data_preprocess(inpath="aclImdb/test/",name="imdb_te.csv", mix=True)
	[Xtest_text,Ytest_polarity ] = retrieve_data(name=test_path, train=True)
	print ("Retrieved the test data. Now will initialize the model \n\n")
	
	glove_word2vec()
	
	vectorizer = tf_process(Xtrain_text)
	Xtrain_uni = vectorizer.transform(Xtrain_text)
	Xtest_uni = vectorizer.transform(Xtest_text)
	Ytest_uni = logistic_regression(Xtrain_uni, Ytrain, Xtest_uni)
	print ("\n\n>> Accuracy of TF and Logistic Regression = ", accuracy(Ytest_polarity, Ytest_uni),"\n\n")
	Ytest_uni = naive_bayes_classifier(Xtrain_uni, Ytrain, Xtest_uni)
	print ("\n\n>> Accuracy of TF and Naive Bayes = ", accuracy(Ytest_polarity, Ytest_uni),"\n\n")
	Ytest_uni = svm_classifier(Xtrain_uni, Ytrain, Xtest_uni)
	print ("\n\n>> Accuracy of TF and SVM Classifier = ", accuracy(Ytest_polarity, Ytest_uni),"\n\n")


	vectorizer = tfidf_process(Xtrain_text)
	Xtrain_uni = vectorizer.transform(Xtrain_text)
	Xtest_uni = vectorizer.transform(Xtest_text)
	Ytest_uni = logistic_regression(Xtrain_uni, Ytrain, Xtest_uni)
	print ("\n\n>> Accuracy of TFIDF and Logistic Regression = ", accuracy(Ytest_polarity, Ytest_uni),"\n\n")
	Ytest_uni = naive_bayes_classifier(Xtrain_uni, Ytrain, Xtest_uni)
	print ("\n\n>> Accuracy of TFIDF and Naive Bayes = ", accuracy(Ytest_polarity, Ytest_uni),"\n\n")
	Ytest_uni = svm_classifier(Xtrain_uni, Ytrain, Xtest_uni)
	print ("\n\n>> Accuracy of TFIDF and SVM Classifier ", accuracy(Ytest_polarity, Ytest_uni),"\n\n")

	vectorizer = bag_of_words(Xtrain_text)
	Xtrain_uni = vectorizer.transform(Xtrain_text)
	Xtest_uni = vectorizer.transform(Xtest_text)
	Ytest_uni = logistic_regression(Xtrain_uni, Ytrain, Xtest_uni)
	print ("\n\n>> Accuracy of Bag Of Words and Logistic Regression = ", accuracy(Ytest_polarity, Ytest_uni),"\n\n")
	Ytest_uni = naive_bayes_classifier(Xtrain_uni, Ytrain, Xtest_uni)
	print ("\n\n>> Accuracy of Bag of Words and Naive Bayes = ", accuracy(Ytest_polarity, Ytest_uni),"\n\n")
	Ytest_uni = svm_classifier(Xtrain_uni, Ytrain, Xtest_uni)
	print ("\n\n>> Accuracy of Bag of words and SVM Classifier ", accuracy(Ytest_polarity, Ytest_uni),"\n\n")


	xtr = glove_vec(Xtrain_text)
	xte = glove_vec(Xtest_text)
	yte = logistic_regression(xtr,Ytrain,xte)
	print("\n\n>> Glove and Logistic",accuracy(Ytest_polarity, yte),"\n\n")
	yte = naive_bayes_classifierG(xtr,Ytrain,xte)
	print("\n\n>> Glove and Naive Bayes ( Gaussian )",accuracy(Ytest_polarity, yte),"\n\n")
	yte = svm_classifier(xtr,Ytrain,xte)
	print("\n\n>> Glove and SVM ",accuracy(Ytest_polarity, yte),"\n\n")
	
	xtr = tfidf_glove_vec(Xtrain_text)
	xte = tfidf_glove_vec(Xtest_text)
	yte = logistic_regression(xtr,Ytrain,xte)
	print("\n\n>> 12Tfidf_GLove and Logistic",accuracy(Ytest_polarity, yte),"\n\n")
	yte = naive_bayes_classifierG(xtr,Ytrain,xte)
	print("\n\n>> 12Tfidf_Glove and Naive Bayes ( Gaussian )",accuracy(Ytest_polarity, yte),"\n\n")
	yte = svm_classifier(xtr,Ytrain,xte)
	print("\n\n>> Tfidf_Glove and SVM ",accuracy(Ytest_polarity, yte),"\n\n")
	
	xtr = word_to_vec(Xtrain_text)
	xte = word_to_vec(Xtest_text)
	yte = logistic_regression(xtr,Ytrain,xte)
	print("\n\n>> Word2Vec and Logistic",accuracy(Ytest_polarity, yte),"\n\n")
	yte = naive_bayes_classifierG(xtr,Ytrain,xte)
	print("\n\n>> Word2Vec and Naive Bayes ( Gaussian )",accuracy(Ytest_polarity, yte),"\n\n")
	yte = svm_classifier(xtr,Ytrain,xte)
	print("\n\n>> Word2Vec and SVM ",accuracy(Ytest_polarity, yte),"\n\n")
	

	xtr = tfidfw2v(Xtrain_text)
	xte = tfidfw2v(Xtest_text)
	yte = logistic_regression(xtr,Ytrain,xte)
	print("\n\n>> Tfidf_Word2vec and Logistic",accuracy(Ytest_polarity, yte),"\n\n")
	yte = naive_bayes_classifierG(xtr,Ytrain,xte)
	print("\n\n>> Tfidf_Word2Vec and Naive Bayes ( Gaussian )",accuracy(Ytest_polarity, yte),"\n\n")
	yte = svm_classifier(xtr,Ytrain,xte)
	print("\n\n>> Tfidf_Word2vec and SVM ",accuracy(Ytest_polarity, yte),"\n\n")
	

	
	

	'''
	print ("-----------------------ANALYSIS ON THE INSAMPLE DATA (TRAINING DATA)---------------------------")
	uni_vectorizer = unigram_process(Xtrain_text)
	print ("Fitting the unigram model")
	Xtrain_uni = uni_vectorizer.transform(Xtrain_text)
	print ("After fitting ")
	#print ("Applying the stochastic descent")
	#Y_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtrain_uni)
	#print ("Done with  stochastic descent")
	#print ("Accuracy for the Unigram Model is ", accuracy(Ytrain, Y_uni))
	print ("\n")

	bi_vectorizer = bigram_process(Xtrain_text)
	print ("Fitting the bigram model")
	Xtrain_bi = bi_vectorizer.transform(Xtrain_text)
	print ("After fitting ")
	#print ("Applying the stochastic descent")
	#Y_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtrain_bi)
	#print ("Done with  stochastic descent")
	#print ("Accuracy for the Bigram Model is ", accuracy(Ytrain, Y_bi))
	print ("\n")

	uni_tfidf_transformer = tfidf_process(Xtrain_uni)
	print ("Fitting the tfidf for unigram model")
	Xtrain_tf_uni = uni_tfidf_transformer.transform(Xtrain_uni)
	print ("After fitting TFIDF")
	#print ("Applying the stochastic descent")
	#Y_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
	#print ("Done with  stochastic descent")
	#print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_uni))
	print ("\n")


	bi_tfidf_transformer = tfidf_process(Xtrain_bi)
	print ("Fitting the tfidf for bigram model")
	Xtrain_tf_bi = bi_tfidf_transformer.transform(Xtrain_bi)
	print ("After fitting TFIDF")
	#print ("Applying the stochastic descent")
	#Y_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
	#print ("Done with  stochastic descent")
	#print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_bi))
	print ("\n")


	print ("-----------------------ANALYSIS ON THE TEST DATA ---------------------------")
	print ("Unigram Model on the Test Data--")
	Xtest_uni = uni_vectorizer.transform(Xtest_text)
	print ("Applying the stochastic descent")
	Ytest_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtest_uni)
	#write_txt(Ytest_uni, name="unigram.output.txt")
	print ("Sto Accuracy for the Unigram Model is ", accuracy(Ytest_polarity, Ytest_uni))
	Ytest_uni = naive_bayes_classifier(Xtrain_uni, Ytrain, Xtest_uni)
	#write_txt(Ytest_uni, name="unigram.output.txt")
	print ("Nav Accuracy for the Unigram Model is ", accuracy(Ytest_polarity, Ytest_uni))
	#Ytest_uni = svm_classifier(Xtrain_uni, Ytrain, Xtest_uni)
	#print ("Svm Accuracy for the Unigram Model is ", accuracy(Ytest_polarity, Ytest_uni))
	Ytest_uni = logistic_regression(Xtrain_uni, Ytrain, Xtest_uni)
	print ("Logistic Accuracy for the Unigram Model is ", accuracy(Ytest_polarity, Ytest_uni))
	print ("\n")


	print ("Bigram Model on the Test Data--")
	Xtest_bi = bi_vectorizer.transform(Xtest_text)
	print ("Applying the stochastic descent")
	Ytest_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtest_bi)
	#write_txt(Ytest_bi, name="bigram.output.txt")
	print ("Sto Accuracy for the Bigram Model is ", accuracy(Ytest_polarity, Ytest_bi))
	Ytest_bi = naive_bayes_classifier(Xtrain_bi, Ytrain, Xtest_bi)
	#write_txt(Ytest_bi, name="bigram.output.txt")
	print ("Nav Accuracy for the Bigram Model is ", accuracy(Ytest_polarity, Ytest_bi))
	#Ytest_bi = svm_classifier(Xtrain_bi, Ytrain, Xtest_bi)
	#print ("Svm Accuracy for the Bigram Model is ", accuracy(Ytest_polarity, Ytest_bi))
	Ytest_bi = logistic_regression(Xtrain_bi, Ytrain, Xtest_bi)
	print ("Logistic Accuracy for the Bigram Model is ", accuracy(Ytest_polarity, Ytest_bi))
	print ("\n")

	print ("Unigram TF Model on the Test Data--")
	Xtest_tf_uni = uni_tfidf_transformer.transform(Xtest_uni)
	print ("Applying the stochastic descent")
	Ytest_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
	#write_txt(Ytest_tf_uni, name="unigramtfidf.output.txt")
	print ("Sto Accuracy for the Unigram TFIDF Model is ", accuracy(Ytest_polarity, Ytest_tf_uni))
	Ytest_tf_uni = naive_bayes_classifier(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
	#write_txt(Ytest_tf_uni, name="unigramtfidf.output.txt")
	print ("Nav Accuracy for the Unigram TFIDF Model is ", accuracy(Ytest_polarity, Ytest_tf_uni))
	#Ytest_tf_uni = svm_classifier(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
	#print ("Svm Accuracy for the Unigram TFIDF Model is ", accuracy(Ytest_polarity, Ytest_tf_uni))
	Ytest_tf_uni = logistic_regression(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
	print ("Logistic Accuracy for the Unigram TFIDF Model is ", accuracy(Ytest_polarity, Ytest_tf_uni))
	print ("\n")

	print ("Bigram TF Model on the Test Data--")
	Xtest_tf_bi = bi_tfidf_transformer.transform(Xtest_bi)
	print ("Applying the stochastic descent")
	Ytest_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
	#write_txt(Ytest_tf_bi, name="bigramtfidf.output.txt")
	print ("Sto Accuracy for the Unigram TFIDF Model is ", accuracy(Ytest_polarity, Ytest_tf_bi))
	Ytest_tf_bi = naive_bayes_classifier(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
	#write_txt(Ytest_tf_bi, name="bigramtfidf.output.txt")
	print ("Nav Accuracy for the Unigram TFIDF Model is ", accuracy(Ytest_polarity, Ytest_tf_bi))
	#Ytest_tf_bi = svm_classifier(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
	#print ("Svm Accuracy for the Unigram TFIDF Model is ", accuracy(Ytest_polarity, Ytest_tf_bi))
	Ytest_tf_bi = logistic_regression(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
	print ("Logistic Accuracy for the Unigram TFIDF Model is ", accuracy(Ytest_polarity, Ytest_tf_bi))
	print ("\n")

	print ("Total time taken is ", time.time()-start, " seconds")
	pass
	'''