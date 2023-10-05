import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

data = pd.read_csv("/Users/buster/Python/gfg/WELFake_Dataset.csv")

# Replace missing values with empty strings
data['text'] = data['text'].fillna('')

x, y = data['text'], data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

x_train_vectorized = vectorizer.fit_transform(x_train)

x_test_vectorized = vectorizer.transform(x_test)

clf = LinearSVC()
clf.fit(x_train_vectorized, y_train)
print(clf.score(x_test_vectorized, y_test))
with open("mytext.txt", "w", encoding="utf-8") as f:
    f.write(x_test.iloc[10])
with open("mytext.txt","r",encoding="utf-8") as f:
    text=f.read()
vectorized_text=vectorizer.transform([text])
print(clf.predict(vectorized_text))
print(y_test.iloc[10])