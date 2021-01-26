import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np


from sklearn import svm
from pprint import pprint 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.model_selection import ShuffleSplit
import pickle



# importing the training dataset
i = 0
myFiles = ['business', 'entertainment', 'politics', 'sport', 'tech']
df = pd.DataFrame(columns=('news', 'category'))
for filename in myFiles:
    folder = os.path.join('C:\\Users\\AA\\AppData\\Local\\Programs\\Python\\Python36\\bbc', filename)
    for file in os.listdir(folder):
        files = os.path.join(folder,file)
#         data = pd.read_csv(files, sep = '  ',header = None, engine = 'python')

        with open(files, 'r') as my_file:
            data = my_file.read()
        
        df.loc[i] = [data, filename]
        i =i+1
        

df.to_csv('file1.csv') 
# print(df.head())






class textCleaning:
	def __init__(self, df):
		self.df = df



	def specialCharacters(self):
		self.df['Content_1'] = self.df['news'].str.replace("\r", " ")
		self.df['Content_1'] = self.df['Content_1'].str.replace("\n", " ")
		self.df['Content_1'] = self.df['Content_1'].str.replace('"', '')




	def upcaseDowncase(self):
		self.df['Content_2'] = self.df['Content_1'].str.lower()






	def punctuation(self):
		punctuation_signs = list("?:!.,;")

		for punct_sign in punctuation_signs:
			self.df['Content_3'] = self.df['Content_2'].str.replace(punct_sign, '')
	

	def pronouns(self):
		self.df['Content_4'] = self.df['Content_3'].str.replace("'s", "")





	def lemmatization(self):
		wordnet_lemmatizer  = WordNetLemmatizer()
		nrows = len(df)
		lemmatized_text_list = []
		for row in range(0, nrows):
			lemmatized_list = []
			text = df.loc[row]['Content_4']
			text_words = text.split(" ")

			for word in text_words:
				lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

			lemmatized_text = " ".join(lemmatized_list)
			lemmatized_text_list.append(lemmatized_text)


		self.df['Content_5'] = lemmatized_text_list





	def stopWords(self):
		stop_words = list(stopwords.words('english'))
		self.df['Content_6'] = self.df['Content_5']

		for stop_word in stop_words:
			regex_stopword = r"\b" + stop_word + r"\b"
			self.df['Content_6'] = self.df['Content_6'].str.replace(regex_stopword, '')

		return self.df









class labelCoding:
	def __init__(self, df):
		self.df = df

	def addLabels(self):
		category_codes = {
		'business': 0,
		'entertainment': 1,
		'politics': 2,
		'sport': 3,
		'tech': 4
		}

		self.df['Category_Code'] = self.df['category']
		self.df = self.df.replace({'Category_Code':category_codes})
		

	def trainTestSplit(self):
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df['Content_Parsed'], self.df['Category_Code'], test_size=0.15, random_state=8)


	def tfidfVectorizer(self):
		ngram_range = (1,2)
		min_df = 10
		max_df = 1.
		max_features = 300

		tfidf = TfidfVectorizer(encoding='utf-8',ngram_range=ngram_range, stop_words=None, lowercase=False, max_df=max_df, min_df=min_df, max_features=max_features, norm='l2', sublinear_tf=True)

		features_train = tfidf.fit_transform(self.X_train).toarray()
		labels_train = self.y_train
		# print(features_train.shape)

		features_test = tfidf.transform(self.X_test).toarray()

		labels_test = self.y_test
		# print(features_test.shape)

		return features_train, labels_train, features_test, labels_test, tfidf





class modelTraining:



	def __init__(self, features_train, labels_train):

		self.features_train = features_train
		self.labels_train = labels_train




	def randomizedSearch(self):



		C = [.0001, .001, .01]
		gamma = [.0001, .001, .01, .1, 1, 10, 100]
		degree = [1, 2, 3, 4, 5]
		kernel = ['linear', 'rbf', 'poly']
		probability = [True]

		random_grid = { 'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree, 'probability': probability }

		# First create the base model to tune
		svc = svm.SVC(random_state=8)

		# Definition of the random search
		random_search = RandomizedSearchCV(estimator=svc,  param_distributions=random_grid,  n_iter=50,  scoring='accuracy', cv=3,   verbose=1,  random_state=8)

		# Fit the random search model
		random_search.fit(self.features_train, self.labels_train)


		return random_search.best_params_, random_search.best_score_, random_search.best_estimator_





	def gridSearch(self):

		C = [.0001, .001, .01, .1]
		degree = [3, 4, 5]
		gamma = [1, 10, 100]
		probability = [True]

		param_grid = [ {'C': C, 'kernel':['linear'], 'probability':probability}, {'C': C, 'kernel':['poly'], 'degree':degree, 'probability':probability}, {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}]


		# Create a base model
		svc = svm.SVC(random_state=8)


		# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
		cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)


		# Instantiate the grid search model
		grid_search = GridSearchCV(estimator=svc, param_grid=param_grid,  scoring='accuracy',  cv=cv_sets,  verbose=1)


		# Fit the grid search to the data
		grid_search.fit(self.features_train, self.labels_train)


		return grid_search.best_params_, grid_search.best_score_, grid_search.best_estimator_



		



	def modelFit(self, best_svc):
		svc_fit = best_svc.fit(self.features_train, self.labels_train)



	def modelPredict(self, best_svc, features_test):
		svc_predict = best_svc.predict(features_test)
		return svc_predict




	def accuracyScore(self, labels_test, svc_predict):
		return accuracy_score(labels_test, svc_predict)










cleaned_text = textCleaning(df)
cleaned_text.specialCharacters()
cleaned_text.upcaseDowncase()
cleaned_text.punctuation()
cleaned_text.pronouns()
cleaned_text.lemmatization()
dataframe = cleaned_text.stopWords()



list_columns = ["category", "news", "Content_6"]
df = dataframe[list_columns]

df = df.rename(columns={'Content_6': 'Content_Parsed'})



encoded_labels = labelCoding(df)
encoded_labels.addLabels()
encoded_labels.trainTestSplit()
features_train, labels_train, features_test, labels_test, tfidf = encoded_labels.tfidfVectorizer()


train_model = modelTraining(features_train, labels_train)
random_param, random_score, random_estimator = train_model.randomizedSearch()
grid_param, grid_score, grid_estimator = train_model.gridSearch()


if grid_score > random_score:

	best_svc = grid_estimator

else:
	best_svc = random_estimator



train_model.modelFit(best_svc)
svc_predict = train_model.modelPredict(best_svc, features_test)
score = train_model.accuracyScore(labels_test, svc_predict)



print(random_score, grid_score, score)




    
# features_train
with open('features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    
# TF-IDF object
with open('tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)


#SVM object
with open('best_svc.pickle', 'wb') as output:
    pickle.dump(best_svc, output)





