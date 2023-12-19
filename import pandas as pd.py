import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

data = {
    'age': [25, 35, 45, 28, 50, 40, 35, 55, 60, 48],
    'job': ['admin', 'technician', 'blue-collar', 'admin', 'management', 'entrepreneur', 'technician', 'retired', 'retired', 'management'],
    'marital': ['single', 'married', 'single', 'single', 'married', 'married', 'divorced', 'married', 'single', 'married'],
    'education': ['secondary', 'tertiary', 'primary', 'tertiary', 'tertiary', 'tertiary', 'secondary', 'secondary', 'tertiary', 'secondary'],
    'default': ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no'],
    'housing': ['yes', 'no', 'yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes'],
    'loan': ['no', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no'],
    'contact': ['cellular', 'telephone', 'cellular', 'cellular', 'telephone', 'cellular', 'cellular', 'telephone', 'cellular', 'telephone'],
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct'],
    'poutcome': ['unknown', 'success', 'unknown', 'failure', 'unknown', 'success', 'unknown', 'unknown', 'unknown', 'unknown'],
    'y': ['no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes']
}

df = pd.DataFrame(data)

label_encoder = LabelEncoder()
df['job'] = label_encoder.fit_transform(df['job'])
df['marital'] = label_encoder.fit_transform(df['marital'])
df['education'] = label_encoder.fit_transform(df['education'])
df['default'] = label_encoder.fit_transform(df['default'])
df['housing'] = label_encoder.fit_transform(df['housing'])
df['loan'] = label_encoder.fit_transform(df['loan'])
df['contact'] = label_encoder.fit_transform(df['contact'])
df['month'] = label_encoder.fit_transform(df['month'])
df['poutcome'] = label_encoder.fit_transform(df['poutcome'])
df['y'] = label_encoder.fit_transform(df['y'])

X = df.drop('y', axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_mat}")
print(f"Classification Report:\n{classification_rep}")
