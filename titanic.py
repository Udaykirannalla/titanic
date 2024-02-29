# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the Titanic dataset
titanic_data = pd.read_csv('C:\\Users\\Sai Sunil\\titanic.csv')


# Preprocessing: handle missing values, encode categorical variables
titanic_data.dropna(subset=['Age', 'Embarked'], inplace=True)  # Drop rows with missing values in 'Age' and 'Embarked'
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})  # Encode 'Sex' feature
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'])  # One-hot encode 'Embarked' feature

# Define features and target variable
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
target = 'Survived'

# Split the data into training and testing sets
X = titanic_data[features]
y = titanic_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline with preprocessing and classification
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('classifier', RandomForestClassifier())  # Classification algorithm
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))