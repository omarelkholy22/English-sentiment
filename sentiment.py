#Imports
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
import numpy as np
from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

#Cleaning the tokens

def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]
def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]
def stemming(text):
    ps = PorterStemmer()
    return [ps.stem(word) for word in text]
def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]
def return_sentences(tokens):
    return " ".join([word for word in tokens])

nltk.download('wordnet')

#Reading data and converting it into one dataframe
training_data = pd.read_csv('twitter_training.csv')
validation_data = pd.read_csv('twitter_validation.csv')
training_data = training_data.drop(training_data.columns[0:2],axis=1)
validation_data = validation_data.drop(validation_data.columns[0:2],axis=1)
training_data.columns = ["label",'tweet']
validation_data.columns = ["label",'tweet']
data = [training_data[:],validation_data[:]]
data = pd.concat(data).reset_index(drop=True)

#the original data had Irrelevant label unlike the desired labels pos,neg,neutral the next two lines drop it
data['label'].replace('Irrelevant', np.nan, inplace=True) 
data.dropna(subset=['label'], inplace=True)


data['tweet'] = data['tweet'].str.replace('\d+','') #removing numbers
data["tweet"] = data["tweet"].str.replace('[^\w\s]','') #removing punctuation and emojis
data['tweet'] = data['tweet'].str.replace('_',' ') #removing underscore '_'
data['tweet'] = data['tweet'].str.replace('  ',' ') #removing the double spaces

#droping the empty cells
data['tweet'].replace('', np.nan, inplace=True)
data.dropna(subset=['tweet'], inplace=True)

#Tokenizing
tweet_tokens = pd.DataFrame(columns=['token','label'])
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
for ind in data.index:
    tweet_tokens = tweet_tokens.append({'token':tokenizer.tokenize(data['tweet'][ind]),'label':data['label'][ind]},ignore_index=True)

#Calling the cleaning functions
tweet_tokens['token'] = tweet_tokens['token'].apply(lambda x : remove_small_words(x))
tweet_tokens['token'] = tweet_tokens['token'].apply(lambda x : remove_stopwords(x))
tweet_tokens['token'] = tweet_tokens['token'].apply(lambda wrd: stemming(wrd))
tweet_tokens['token'] = tweet_tokens['token'].apply(lambda x : lemmatize(x))
tweet_tokens['clean_text'] = tweet_tokens['token'].apply(lambda x : return_sentences(x))

#converting labels to numbers
tweet_tokens['label'].replace({'Positive':1,'Negative':0,'Neutral':2},inplace=True)

#Creating the word vector using tfidf
X_train, X_test, y_train, y_test = train_test_split(tweet_tokens['clean_text'], tweet_tokens['label'], test_size = 0.2)
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

#used randomforest as it gave the best accuracy among LogisticRegression, XGBClassifier, LGBMClassifier
randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)
print("Accuracy score : {}".format(accuracy_score(y_test, y_pred)))
print("Confusion matrix : \n {}".format(confusion_matrix(y_test, y_pred)))
