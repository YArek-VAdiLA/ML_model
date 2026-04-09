import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('Antiplagiarism.csv')
print(df.info())
print(df.duplicated())
print(len(df[df['generated'] == 0]))
print(len(df[df['generated'] == 1]))
df = df.drop_duplicates()
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

vector = TfidfVectorizer(max_features= 100, ngram_range=(1,2))
X_train_vector = vector.fit_transform(X_train)
X_test_vector = vector.transform(X_test)

model = RandomForestClassifier(
    n_estimators= 200,
    max_depth = 5,
    criterion = 'log_loss',
    random_state= 42,
    max_features='sqrt'
)

model.fit(X_train_vector,y_train)
pred = model.predict(X_test_vector)
acc = accuracy_score(y_test,pred)
print(acc)