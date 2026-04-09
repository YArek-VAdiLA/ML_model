import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('Spam_or_not_spam.csv')
print(df.head())

X = df['text']
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size= 0.2,
    random_state= 42,
    stratify= y
)
vector = TfidfVectorizer(max_features= 1000, ngram_range=(1,2))
X_train_vector = vector.fit_transform(X_train)
X_test_vector = vector.transform(X_test)

model = KNeighborsClassifier(
    n_neighbors= 5,
    weights='uniform',
    algorithm='auto',
    p=2
)
model.fit(X_train_vector,y_train)
pred = model.predict(X_test_vector)
acc = accuracy_score(y_test, pred)
print(acc)
