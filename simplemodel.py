from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import dataprocessing
import joblib

filename = "10k_merged_dataset.csv"
raw_data = dataprocessing.get_data(filename)
cleaned_data = raw_data.copy()
cleaned_data['content'] = cleaned_data['content'].apply(dataprocessing.text_preprocessing)

# Convert the 'type' column to binary labels
cleaned_data['label'] = cleaned_data['type'].apply(lambda x: 1 if x == 'reliable' else 0)

# Join the tokenized words back into a string
cleaned_data['content'] = cleaned_data['content'].apply(' '.join)

# Initialize TfidfVectorizer
vectorizer = CountVectorizer()

# Fit and transform the 'content' column
X = vectorizer.fit_transform(cleaned_data['content'])

# Get the labels
y = cleaned_data['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'simple_model.pkl')
joblib.dump(vectorizer, 'simple_vectorizer.pkl')

load_model = joblib.load('simple_model.pkl')
load_vectorizer = joblib.load('simple_vectorizer.pkl')

# Make predictions
y_pred = load_model.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)

print("Model accuracy: ", acc)