import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder



df = pd.read_csv('Credit_scoring.csv')
df_t = df.transpose()
print(df_t.head)

cate_columns = df.select_dtypes(include=['object', 'str']).columns
for col in cate_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(columns='loan_status')
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state= 42,
    stratify= y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

KNN = KNeighborsClassifier(
    n_neighbors=7,
    weights='uniform',
    algorithm='auto',
    p=2
)

KNN.fit(X_train_scaled, y_train)
pred = KNN.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)
print(acc)

