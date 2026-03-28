import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load data
df = pd.read_csv("credit_risk.csv")

# encode
le = LabelEncoder()
df["Default"] = le.fit_transform(df["Default"])

# split
X = df.drop("Default", axis=1)
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# model
model = LogisticRegression()
model.fit(X_train,y_train)

# predict
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,pred))