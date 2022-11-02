# Write the script to preprocess data, train and evaluate your model here.
# In[1]: imports

import pandas as pd
import numpy as np
import unidecode
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer # (converter as colunas em um array)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


# Function
def reducing_incorrect_character_repeatation(text):
    """
    This Function will reduce repeatition to two characters 
    for alphabets and to one character for punctuations.
    
    arguments:
         input_text: "text" of type "String".
         
    return:
        value: Finally formatted text with alphabets repeating to 
        two characters & punctuations limited to one repeatition 
        
    Example:
    Input : Realllllllllyyyyy,        Greeeeaaaatttt   !!!!?....;;;;:)
    Output : Reallyy, Greeaatt !?.;:)
    
    """
    # Pattern matching for all case alphabets
    Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
    
    # Limiting all the  repeatation to two characters.
    Formatted_text = Pattern_alpha.sub(r"\1\1", text) 
    
    # Pattern matching for all the punctuations that can occur
    Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
    
    # Limiting punctuations in previously formatted string to only one.
    Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
    
    # The below statement is replacing repeatation of spaces that occur more than two times with that of one occurrence.
    Final_Formatted = re.sub(' {2,}',' ', Combined_Formatted)
    return Final_Formatted


# In[2]:

# Importing
df_sp = pd.read_csv('dataset-spanish.csv', header = 0)
df_hi = pd.read_csv('dataset-hindi.csv', header = 0)

# Data Viz
df_sp.sample(5)
df_hi.sample(5)

# Checking the dataset lenght
print(f'Spanish dataset length: {len(df_sp)} \nHindi dataset lenght: {len(df_hi)}')

# Describe
df_sp.describe()
df_hi.describe()


# In[3]: Data Cleaning
# Before starting the data cleaning process, I'll add the column 'Idiom' in the datasets so I can already concatenate them to make the cleaning for the whole dataset


df_sp['Idiom'] = 'Spanish'
df_hi['Idiom'] = 'Hindi'
df = pd.concat([df_sp, df_hi])
print(len(df))


# Duplicated rows:
print(df.duplicated().sum())
df.drop_duplicates(inplace= True)


# Missing values
df.dropna(inplace=True)


# Checking dataset lenght after dropping the duplications for the spanish and the hindi (so we can see if it's still balanced)
print(f"Spanish dataset length: {len(df[(df.Idiom == 'Spanish')])} \nHindi dataset lenght: {len(df[(df.Idiom == 'Hindi')])}")


# Taking a deep look into the variables to clean it properly
df.describe()
df['song_name'].value_counts()

# I'll remove the rows with Title Error (and TITLE_ERROR) and also the Intro songs, since it doesn't mean much about the idiom of the song (which is what we are looking for in this project)
df = df[(df.song_name != 'Title Error') & (df.song_name != 'TITLE_ERROR') & (df.song_name.str.startswith('Intro') == False)]


# Removing possible double whitespace in the text
for i in range(0,len(df.columns)):
    for j in range(0, len(df)):
        df[(df.columns.values[i])].values[j] = re.sub(r'\s+', ' ', df[(df.columns.values[i])].values[j])
        

# Visualizing
df.sample(5)


# In[4]: Pre Processing

# Removing accented characters
for i in range(0,len(df.columns)):
    for j in range(0, len(df)):
        df[(df.columns.values[i])].values[j] = unidecode.unidecode(df[(df.columns.values[i])].values[j])
        


# Converting the dataset to lowercase
for i in range(0,len(df.columns)):
    for j in range(0, len(df)):
        df[(df.columns.values[i])].values[j] = (df[(df.columns.values[i])].values[j]).lower()
        
        

# Reducing repeated characters and punctuations
for i in range(0,len(df.columns)):
    for j in range(0, len(df)):
        df[(df.columns.values[i])].values[j] = reducing_incorrect_character_repeatation(df[(df.columns.values[i])].values[j])
        
        
    
# Removing Special Characters and also numbers(since the numbers doesn't give any hint regarding the idiom)
for i in range(0,len(df.columns)):
    for j in range(0, len(df)):
        df[(df.columns.values[i])].values[j] = re.sub("[.,;/0-9!?¿%@|~'#+:()-]", '', df[(df.columns.values[i])].values[j])



# Visualizing data 
df.sort_values('song_name')


# Remove the rows where we got empty values after the pre-processing and couldn't be removed by .dropna()
df = df[(df.song_name != '') & (df.release_name != '') & (df.artist_name != '')]


# Visualizing
df.sample(5)


# We could see that some albuns had the 'vol' in its end (indicating which volum it is). Since it's a generic word, it won't help to keep them in our dataset. Let's remove the word then.
for i in range(0, len(df)):
    if df.release_name.str.endswith('vol ').values[i]:
        df.release_name.values[i] = df.release_name.values[i].replace('vol ', '')
    elif df.release_name.str.endswith('vol').values[i]:
        df.release_name.values[i] = df.release_name.values[i].replace('vol', '')
        
        
# Transforming the target variable ('Idiom') in 0 and 1 ("0" -> 'hindi ; "1" -> spanish)
df['Idiom'] = np.where((df['Idiom']) == 'spanish', 1,0)

print(df['Idiom'].value_counts())


# In[5]: Machine Learning

# Firstly, I'll use all the 3 variables as target. Later I'll try using only the song_name (since the we have repeated values in the album name and artist, and also artist name do not neceesary mean anything about the idiom)

# Train and Test dataset
x = (df.drop(['Idiom'], axis = 1)) # predictors
y = (df['Idiom']) # target

# Transforming the variables to use in our algorith
cv = CountVectorizer()
ct = make_column_transformer((cv,'artist_name'),(cv,'song_name'),(cv,'release_name'))
X = ct.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)


# Verifying if the variable is balanced
plt.figure(figsize = (8,6))
ax = sns.countplot(y_train)


# Training model
model = MultinomialNB()
model.fit(X_train, y_train)


# In[6]:Score
score = model.score(X_test, y_test)
print(score)


# Confusion Matrix
predictions = model.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))


# Precision
print(metrics.classification_report(y_test, predictions))


# I'll be testing the same model, but now using only the "song_name" as predictor variable. This one might be a more generic model.

# In[6]:Model 2
x = np.array((df['song_name'])) # predictors
y = np.array((df['Idiom'])) # target

cv = CountVectorizer()
X = cv.fit_transform(x)

model1 = MultinomialNB()
model1.fit(X_train, y_train)

# In[62]: Score 2

score = model1.score(X_test, y_test)
print(score)

predictions = model1.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))

# We got a worse result using only the 'song_name' variable. But it actually might be better for futures datasets, since for this one we have a lot of repeated values for albuns and artists name.

# In[24]: KNN

x = (df.drop(['Idiom'], axis = 1)) # predictors
y = (df['Idiom']) # target

cv = CountVectorizer()
ct = make_column_transformer((cv,'artist_name'),(cv,'song_name'),(cv,'release_name'))
X = ct.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)


model2 = KNeighborsClassifier()
model2.fit(X_train, y_train)


score = model2.score(X_test, y_test)
print(score)

predictions = model2.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))


# Now using only song_name as predictor

# In[25]:Model 4

x = np.array((df['song_name'])) # predictors
y = np.array((df['Idiom'])) # target

cv = CountVectorizer()
X = cv.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# Trainning model
model3 = KNeighborsClassifier()
model3.fit(X_train, y_train)


score = model3.score(X_test, y_test)
print(score)

predictions = model3.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))

# In[ ]:# Conclusion

# We end up having a very good accuracy with the models. Altough the Naive Bayes model had a better perfomance than the KNN one. Also the result was higher when all the variables were used.  

# Even though there were some repeated values for 'artist_name' and 'release_name' , I don't think it'll affect negatively our result, since the name is usually writen with the language the song is.




