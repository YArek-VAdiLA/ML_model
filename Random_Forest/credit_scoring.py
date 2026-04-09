import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


df  = pd.read_csv('Credit_scoring.csv')
print(df.info())

cate_colums = df.select_dtypes(include=['object', 'str']).columns
for cl in cate_colums:
    le = LabelEncoder()
    df[cl] = le.fit_transform(df[cl])

X = df.drop(columns='loan_status')
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size= 0.2,
    random_state= 42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators= 400,
    max_depth= 5,
    criterion='log_loss',
    random_state=42,
    max_features='sqrt'
)
model.fit(X_train,y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(acc)