
# coding: utf-8

# In[42]:


import csv
import pandas as pd
from textblob import TextBlob as TB
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.linear_model import LinearRegression 
import matplotlib.gridspec as gridspec 
import matplotlib.gridspec as gridspec 
import seaborn as sns 
import time

#settings
import warnings
import plotly.offline as pyo
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
warnings.filterwarnings("ignore")
pyo.init_notebook_mode()


get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


df = pd.read_csv("yelp_academic_dataset/yelp_review.csv",  sep=",",parse_dates=['date'])


# In[ ]:


polarity=list()
subjectivity_list = list()
sentiment_list = list()

for row in df["text"]:
    sentences = str(row)
    blob = TB(sentences)
    polarity.append(blob.sentiment.polarity)
    subjectivity_list.append(blob.sentiment.subjectivity)
    
    if blob.sentiment.polarity == 0:
        X="neutral" 
        sentiment_list.append("neutral")
    elif blob.sentiment.polarity < 0:
        X="negative"
        sentiment_list.append("negative")
    elif blob.sentiment.polarity > 0:
        X="positive"
        sentiment_list.append("positive")
         
df["Polarity"]=polarity
df["sentiment_list"]=sentiment_list
df["subjectivity_list"]=subjectivity_list

df.head()
df.to_csv('yelp.csv', index=False)


# In[15]:


df1 = df.sample(frac=0.1, replace=True)


# In[2]:


Polarity=df1["Polarity"]
stars= df1["stars"]
useful= df1["useful"]
funny = df1["funny"]
cool = df1["cool"]


# In[3]:


from scipy.stats import linregress
linregress(Polarity,stars)


# In[4]:


linregress(Polarity,useful)


# In[5]:


linregress(Polarity,funny)


# In[6]:


linregress(Polarity,cool)


# In[94]:


df1.to_csv('yeplex.csv', index=False)


# In[111]:


df2 = df.sample(frac=0.001, replace=True)


# In[112]:


Polarity2=df2["Polarity"]
stars2= df2["stars"]


# In[113]:


lr = LinearRegression()


# In[114]:


from scipy.stats import linregress
import numpy as np
Polarity2 =Polarity2.values.reshape(-1, 1)
stars2 =stars2.values.reshape(-1, 1)


# In[115]:


lr.fit(Polarity2,stars2)


# In[116]:


print(lr.intercept_)


# In[117]:


print(lr.coef_)


# In[118]:


fig, ax = plt.subplots()
fig.clf()
ax = fig.add_subplot(111)
ax.set_xlabel("Polarity")
ax.set_ylabel("stars")
ax.scatter(Polarity2,stars2)
xmin, xmax = ax.get_xlim()
ax.plot([xmin, xmax], [lr.predict(xmin)[0][0], lr.predict(xmax)[0][0]], linewidth=3, color="red")
ax.set_xlim([xmin, xmax])
fig


# In[8]:


df['text length'] = df['text'].apply(len)
df.head()


# In[11]:


sns.boxplot(x='stars', y='text length', data=df)


# In[12]:


stars = df.groupby('stars').mean()
stars.corr()


# In[13]:


sns.heatmap(data=stars.corr(), annot=True)


# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
cv = CountVectorizer()
df3 = pd.read_csv("yelp_academic_dataset/yelp_review.csv",  sep=",",parse_dates=['date'])


# In[3]:


df4 = df3.sample(frac=0.001, replace=True)


# In[4]:


len(df4)


# In[5]:


df5=df4[(df4['stars']==1)|(df4['stars']==3)|(df4['stars']==5)]


# In[6]:


#classification
x = df5['text']
y = df5['stars']
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)


# In[7]:


from nltk.corpus import stopwords
import string
def text_clean(message):
    nopunc = [i for i in message if i not in string.punctuation]
    nn = "".join(nopunc)
    nn = nn.lower().split()
    nostop = [words for words in nn if words not in stopwords.words('english')]
    return(nostop)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
cv_transformer = CountVectorizer(analyzer = text_clean)


# In[9]:


x = cv_transformer.fit_transform(x)


# In[10]:


x.shape


# In[11]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)


# In[12]:


#Training a Naive bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(x_train, y_train)


# In[13]:


#predictions
predictions= nb.predict(x_test)
predictions


# In[14]:


#Creating a confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print("\n")
print(classification_report(y_test, predictions))


# In[15]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='gini')
rf.fit(x_train, y_train)


# In[16]:


pred_rf = rf.predict(x_test)


# In[17]:


print("Confusion Matrix\n",confusion_matrix(y_test, pred_rf))
print("\n")
print("Classification report\n",classification_report(y_test, pred_rf))


# In[18]:


from sklearn import linear_model
lg = linear_model.LogisticRegression(C=1.5)
lg.fit(x_train, y_train)


# In[19]:


pred_lg = lg.predict(x_test)


# In[20]:


print(confusion_matrix(y_test,pred_lg ))
print("\n")
print(classification_report(y_test,pred_lg))


# In[28]:


user_agg=df.groupby('user_id').agg({'review_id':['count'],'date':['min','max'],
                                'useful':['sum'],'funny':['sum'],'cool':['sum'],
                               'stars':['mean']})
user_agg=user_agg.sort_values([('review_id','count')],ascending=False)


# In[43]:


# Cap max reviews to 30 for better visuals
user_agg[('review_id','count')].loc[user_agg[('review_id','count')]>30] = 30
plt.figure(figsize=(12,5))
plt.suptitle("User Deep dive\n",fontsize=20)
gridspec.GridSpec(1,2)
plt.subplot2grid((1,2),(0,0))
#Cumulative Distribution
ax=sns.kdeplot(user_agg[('review_id','count')],shade=True,color='r')
plt.title("How many reviews does an average user give?",fontsize=15)
plt.xlabel('# of reviews given', fontsize=12)
plt.ylabel('# of users', fontsize=12)

#Cumulative Distribution
plt.subplot2grid((1,2),(0,1))
sns.distplot(user_agg[('review_id','count')],
             kde_kws=dict(cumulative=True))
plt.title("Cumulative dist. of user reviews",fontsize=15)
plt.ylabel('Cumulative perc. of users', fontsize=12)
plt.xlabel('# of reviews given', fontsize=12)

plt.show()
end_time=time.time()
print("Took",end_time-start_time,"s")


# In[39]:


selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 
                  'awful', 'wow', 'hate']
selected_words


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=selected_words, lowercase=False)
#corpus = ['This is the first document.','This is the second second document.']
#print corpus
selected_word_count = vectorizer.fit_transform(tip['text'].values.astype('U'))
vectorizer.get_feature_names()


# In[27]:


word_count_array = selected_word_count.toarray()
word_count_array.shape


# In[28]:


word_count_array.sum(axis=0)


# In[29]:


temp = pd.DataFrame(index=vectorizer.get_feature_names(),                     data=word_count_array.sum(axis=0)).rename(columns={0: 'Count'})
temp.reset_index(level=0, inplace=True)


# In[30]:


temp.values


# In[34]:


d = {}
for a, x in temp.values:
    d[a] = x

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(width=600, height=400, random_state=1, max_words=200000000)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure(figsize=(11,6), facecolor='k')
#plt.title("Tips for top reviewed restaurant", fontsize=40,color='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[35]:


# Generate training data
train_labels = []
train_data = reviews['text'].head(2000).tolist()
for n in reviews['stars'].head(2000).as_matrix():
    res = 'pos' if n > 3 else 'neg'
    train_labels.append(res)

# Generate testing data
test_labels = []
test_data = reviews['text'].tail(2000).tolist()
for n in reviews['stars'].tail(2000).as_matrix():
    res = 'pos' if n > 3 else 'neg'
    test_labels.append(res)


# In[36]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(train_data)


# In[37]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)


# In[39]:


##Testing the Performance
test_counts = count_vect.transform(test_data)
test_tfidf = tfidf_transformer.transform(test_counts)
predicted = clf.predict(test_tfidf)
print('%.1f%%' % (np.mean(predicted == test_labels) * 100))training the classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_tfidf, train_labels)


# In[40]:


docs = ['this place is good', 'this place is terrible']
counts = count_vect.transform(docs)
tfidf = tfidf_transformer.transform(counts)
predicted = clf.predict(tfidf)
for doc, category in zip(docs, predicted):
    print('%r => %s' % (doc, category))


# In[41]:


reviews['review_length'] = reviews.text.map(len)


# In[42]:


import seaborn as sns
sns.set()
# check how the review lengths are distributed
ax = sns.FacetGrid(data=reviews, col='stars', xlim=(0, 2000)).map(plt.hist, 'review_length', bins=50)
ax.axes[0][0].set(ylabel='number of reviews');


# In[43]:


stval = reviews.groupby('stars').mean()
stval


# In[44]:


Cust = []
for i in reviews['stars']:
    if (i == 1):
        Cust.append('BAD')
    elif (i == 3) | (i == 2):
        Cust.append('NEUTRAL')
    else:
        Cust.append('GOOD')
        

reviews['Customer EXP'] = Cust
reviews['Customer EXP'].value_counts()
reviews['Text length'] = reviews['text'].apply(lambda x:len(x.split()))
reviews.head()


# In[45]:


#exploratory data analysis
a = sns.FacetGrid(data = reviews, col = 'Customer EXP', hue = 'Customer EXP', palette='plasma', size=5)
a.map(sns.distplot, "Text length")
reviews.groupby('Customer EXP').mean()['Text length']

