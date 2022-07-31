# pip install pandas
import pandas as pd
# pip install scikit-learn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("C:\\Users\\Asus\\Desktop\\New folder\\emails.csv")
df['label'] = df.spam.map({0:'ham',1:'spam'})

X = df.text
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# reference : https://www.studytonight.com/post/scikitlearn-countvectorizer-in-nlp
vect = CountVectorizer()
X_train_count = vect.fit_transform(X_train)
X_test_count = vect.transform(X_test)

# reference : https://www.mygreatlearning.com/blog/multinomial-naive-bayes-explained/
sd = MultinomialNB()
sd.fit(X_train_count,y_train)

predictions = pd.Series(sd.predict(X_test_count))
print(f"predicted output : \n{predictions}\n")
print(f"actual output : \n{y_test}")

score = accuracy_score(y_test, predictions)
print(f"accuracy : {score}")

# add this only if they ask for custom input
# give input "Hey Whatsapp" for ham
# give input "lottery" for spam
# import numpy as np
# t = [input("Enter the subject of the emain : ")]
# t = np.array(t)
# t = vect.transform(t)
# prediction = sd.predict(t)
# print(prediction)
