
import pandas as pd 
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

df= pd.read_csv(r"C:\Users\Anita\Desktop\Model\spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.head()
#rename the columns 
df=df.rename(columns = {"v1": "target", "v2": "text"})
df.head()
#creating another column named label that will transform values in target column to 0 and 1 

df['label'] = df['target'].map({'ham':0,'spam':1})
df.head()

X = df['text']
Y = df['label']
cv = CountVectorizer()

X = cv.fit_transform(X) 


#splitting dataset to train and test. test size=0.3
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=123)
# naive bayes  model call
model = MultinomialNB()
model.fit(X_train, Y_train)
model.score(X_test,Y_test)
y_pred = model.predict(X_test)
print(classification_report(Y_test, y_pred))

#pickle.dump(model, open('newmodel.pkl', 'wb')) #saving model as a pickle file 
joblib.dump(model, "SpamHam_model.pkl")
joblib.dump(cv, "cv.pkl")








