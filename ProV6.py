#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd 
import re,string
import nltk 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# A - Analyse exploratoire

# In[15]:


perc = (gop.isna().sum()/len(gop)*100).round(1)
misscount = gop.isna().sum()
missingtbl = pd.concat([perc.to_frame(), misscount.to_frame()], axis = 1)
missingtbl.columns = ['perc', 'misscount']
missingtbl


# In[16]:


print(gop.relevant_yn.value_counts())
fig, axs = plt.subplots(2,figsize=(15,15))
gop.relevant_yn_confidence.value_counts(bins = 10, normalize = True).plot(kind = 'bar', ax = axs[0])
gop.sentiment_confidence.value_counts(bins = 10, normalize = True).plot(kind = 'bar', ax = axs[1])


# In[17]:


fig, axs = plt.subplots(2,figsize=(15,15))
gop.candidate.value_counts().plot(kind='pie', autopct=lambda 
                                  p: '{:1.0f}%({:.0f})'.format(p,(p/100)*gop.groupby('candidate').size().sum()), ax = axs[0])
gop.subject_matter.value_counts().plot(kind='pie', autopct=lambda 
                                  p: '{:1.0f}%({:.0f})'.format(p,(p/100)*gop.groupby('subject_matter').size().sum()), ax = axs[1])


# In[19]:


gop_corr = gop[['candidate','subject_matter']]
gop_corr = pd.get_dummies(gop_corr)
gop_corr
gop_corr = gop_corr[gop_corr.columns].corr()
plt.figure(figsize = (30,20))
import seaborn as sns
sns.heatmap(gop_corr, annot = True, linewidths=0.2, fmt=".2f")


# In[20]:


gop.sentiment.value_counts().plot(kind='pie', autopct=lambda 
                                  p: '{:1.0f}%({:.0f})'.format(p,(p/100)*gop.groupby('sentiment').size().sum()),
                                  colors=["red", "yellow", "green"])


# In[21]:


gops = gop.groupby(['candidate', 'sentiment']).sentiment.count().unstack()
gops.plot(kind='bar')


# B - Traitement des tweets

# In[2]:


gop = pd.read_csv('sentiment.csv')
tweet=gop['text']
sentiment=gop['sentiment']
def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)
tweet=tweet.str.replace('(RT)','')
tokenizer=RegexpTokenizer(r"\w+")
newtweet=tweet.apply(lambda x: tokenizer.tokenize(x.lower()))
def remove_stopwords(text):
    words=[w for w in text if w not in stopwords.words('english')]
    return words
newtweet=newtweet.apply(lambda x: remove_stopwords(x))
lemmatizer=WordNetLemmatizer()
def word_lemmatizer(text):
    lem=" ".join([lemmatizer.lemmatize(i) for i in text])
    return lem
newtweet=newtweet.apply(lambda x: word_lemmatizer(x))
newtweet


# In[9]:


pos=gop[gop['sentiment']=='Positive']
pos=pos['text']
neg=gop[gop['sentiment']=='Negative']
neg=neg['text']
neuu=gop[gop['sentiment']=='Neutral']
neuu=neuu['text']

from wordcloud import WordCloud,STOPWORDS
def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                            and not word.startswith('@')
                            and not word.startswith('#')])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Neutral words")
wordcloud_draw(neuu)


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer (max_features=2500) 
newtweet = vectorizer.fit_transform(newtweet).toarray()


# In[4]:


gop['newsentiment']=gop['sentiment'].replace({'Negative':-1 ,'Neutral':0,'Positive':1})
newsentiment=np.array(gop['newsentiment'])


# C - Train Test Split

# In[5]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(newtweet, newsentiment, test_size=0.2, random_state=0)


# SVC

# In[11]:


from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

accuracy = []
wd=[]
k = range(1, 300, 1)
for i in k:
    j = i/100
    modelSVC = LinearSVC(penalty="l2",C=j, dual=False,tol=1e-4)
    modelSVC.fit(Xtrain,ytrain)
    ypredSVC = modelSVC.predict(Xtest)
    accuracy.append(accuracy_score(ytest, ypredSVC))
    wd.append(i/100)
plt.plot(wd,accuracy)


# In[12]:


from sklearn.svm import LinearSVC
modelSVC = LinearSVC(penalty="l2",C=0.4, dual=False,tol=1e-4)
modelSVC.fit(Xtrain,ytrain)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
ypredSVC = modelSVC.predict(Xtest)
labels = np.unique(ytest)
print(pd.DataFrame(confusion_matrix(ytest,ypredSVC), columns = labels, index = labels ))
print('\n')
print(classification_report(ytest, ypredSVC))
print('accuracy score = {:.2f}'.format(accuracy_score(ytest, ypredSVC)))


# KNN

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
accuracy = []
k = range(1,500,50)
for i in k :
    modelKNN = KNeighborsClassifier(i, weights = 'uniform')
    modelKNN.fit(Xtrain,ytrain)
    ypredKNN = modelKNN.predict(Xtest)
    accuracy.append(accuracy_score(ytest, ypredKNN))

plt.plot(k,accuracy)


# In[28]:


from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(150, weights = 'uniform')
modelKNN.fit(Xtrain,ytrain)
ypredKNN = modelKNN.predict(Xtest)

print(pd.DataFrame(confusion_matrix(ytest,ypredKNN), columns = labels, index = labels ))
print()
print(classification_report(ytest, ypredKNN))
print()
print('accuracy score = {:.2f}'.format(accuracy_score(ytest, ypredKNN)))


# Random forest

# In[5]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)


# In[4]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 1, stop = 200, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 1500, num = 100)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
max_leaf_nodes = [int(x) for x in np.linspace(10, 600, num = 20)]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'max_leaf_nodes': max_leaf_nodes}


# In[ ]:


rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(Xtrain, ytrain)
rf_random.best_params_


# In[11]:


from sklearn.ensemble import RandomForestClassifier
text_classifier = RandomForestClassifier(n_estimators=60, max_depth=1200, min_samples_split=10, max_features=300, max_leaf_nodes=450)
text_classifier.fit(Xtrain, ytrain)
predictionstest = text_classifier.predict(Xtest)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#predictionstrain = text_classifier.predict(Xtrain)
#print(confusion_matrix(ytrain,predictionstrain))
#print(classification_report(ytrain,predictionstrain))
#print(accuracy_score(ytrain, predictionstrain))

print(confusion_matrix(ytest,predictionstest))
print()
print(classification_report(ytest,predictionstest))
print()
print('accuracy score = {:.2f}'.format(accuracy_score(ytest, predictionstest)))


# Naive Bayes Multinomial

# In[14]:


from sklearn.naive_bayes import MultinomialNB
modelC = MultinomialNB()
modelC.fit(Xtrain, ytrain)
ypredtest = modelC.predict(Xtest)
from sklearn import metrics
print(confusion_matrix(ytest,ypredtest))
print(metrics.accuracy_score(ytest, ypredtest))
print(classification_report(ytest,ypredtest))
#ypredtrain = modelC.predict(Xtrain)
#print(confusion_matrix(ytrain, ypredtrain))
#print(metrics.accuracy_score(ytrain, ypredtrain))
#print(classification_report(ytrain, ypredtrain))


# D- Cross Validation / Logistic Regression

# In[12]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tvec = TfidfVectorizer(stop_words=None, max_features=100000, ngram_range=(1, 3))
lr = LogisticRegression()
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

def lr_cv(splits, X, Y, pipeline, average_method):
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train, test in kfold.split(X, Y):
        lr_fit = pipeline.fit(X[train], Y[train])
        prediction = lr_fit.predict(X[test])
        scores = lr_fit.score(X[test],Y[test])
        accuracy.append(scores * 100)
        precision.append(precision_score(Y[test], prediction, average=average_method)*100)
        print('              negative    neutral     positive')
        print('precision:',precision_score(Y[test], prediction, average=None))
        recall.append(recall_score(Y[test], prediction, average=average_method)*100)
        print('recall:   ',recall_score(Y[test], prediction, average=None))
        f1.append(f1_score(Y[test], prediction, average=average_method)*100)
        print('f1 score: ',f1_score(Y[test], prediction, average=None))
        print('-'*50)
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
from sklearn.pipeline import Pipeline
original_pipeline = Pipeline([
    ('vectorizer', tvec),
    ('classifier', lr)
])


# Imbalanced Data

# In[13]:


lr_cv(3, gop.text, gop.sentiment, original_pipeline, 'macro')


# Random OverSampler

# In[14]:


from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
ROS_pipeline = make_pipeline(tvec, RandomOverSampler(random_state=777),lr)
lr_cv(3, gop.text, gop.sentiment, ROS_pipeline, 'macro')


# SMOTE

# In[15]:


SMOTE_pipeline = make_pipeline(tvec, SMOTE(random_state=777),lr)
lr_cv(3, gop.text, gop.sentiment, SMOTE_pipeline, 'macro')


# In[ ]:




