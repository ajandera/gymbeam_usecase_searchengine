import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Load the CSV dataset using pandas
df = pd.read_csv('products.csv')

# Preprocess string data
# For simplicity, let's assume 'description' is a string column and we will drop it for this example
df.drop(columns=['název', 'popis', 'recenze'], inplace=True)
print(df)
# Split the dataset into features and labels
X = df.drop(columns=['kategorie'])  # Features
y = df['kategorie']  # Labels

# Encode categorical labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the decision tree classifier using Scikit-learn
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
