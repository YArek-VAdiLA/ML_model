import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Antiplagiarism.csv')
df = df.drop_duplicates()
print(df.head(5))
print(df.info())
print(len(df[df['generated'] == 0]))
print(len(df[df['generated'] == 1]))


X = df['text']
y = df['generated']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X_train_vectorized = vectorizer.fit_transform(X_train)

X_test_vectorized = vectorizer.transform(X_test)

model = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    p=2
)

model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)

acc = accuracy_score(y_test, y_pred)
print(acc)
pc = precision_score(y_test, y_pred)
print(pc)
cm = confusion_matrix(y_test, y_pred)
print(cm)


