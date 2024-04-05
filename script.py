import pandas as pd
import dataprocessing
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

advanced_model = joblib.load('advanced_model.pkl')
advanced_vectorizer = joblib.load('advanced_vectorizer.pkl')
simple_model = joblib.load('simple_model.pkl')
simple_vectorizer = joblib.load('simple_vectorizer.pkl')

def preprocess_data(data):
    data['content'] = data['content'].apply(dataprocessing.text_preprocessing)

    data['label'] = data['type'].apply(lambda x: 1 if x == 'reliable' else 0)

    data['content'] = data['content'].apply(' '.join)
    return data

def preprocess_liar(data):
    data['content'] = data['content'].apply(dataprocessing.text_preprocessing)

    data['label'] = data['type'].apply(lambda x: 1 if x == 'true' else 0)

    data['content'] = data['content'].apply(' '.join)
    return data

def evaluate_advanced_merged(new_data):

    preprocessed_new_data = preprocess_data(new_data)

    X_new = advanced_vectorizer.transform(preprocessed_new_data['content'])

    y_new = preprocessed_new_data['label']

    y_pred = advanced_model.predict(X_new)

    f_score = f1_score(y_new, y_pred)

    acc = accuracy_score(y_new, y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_new, y_pred)

    return acc, f_score, cm

def evaluate_simple_merged(new_data):

    preprocessed_new_data = preprocess_data(new_data)

    X_new = simple_vectorizer.transform(preprocessed_new_data['content'])

    y_new = preprocessed_new_data['label']

    y_pred = simple_model.predict(X_new)
    
    f_score = f1_score(y_new, y_pred)

    acc = accuracy_score(y_new, y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_new, y_pred)

    return acc, f_score, cm

def evaluate_advanced_liar(new_data):

    preprocessed_new_data = preprocess_liar(new_data)

    X_new = advanced_vectorizer.transform(preprocessed_new_data['content'])

    y_new = preprocessed_new_data['label']

    y_pred = advanced_model.predict(X_new)

    f_score = f1_score(y_new, y_pred)

    acc = accuracy_score(y_new, y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_new, y_pred)

    return acc, f_score, cm

def evaluate_simple_liar(new_data):

    preprocessed_new_data = preprocess_liar(new_data)

    X_new = simple_vectorizer.transform(preprocessed_new_data['content'])

    y_new = preprocessed_new_data['label']

    y_pred = simple_model.predict(X_new)
    
    f_score = f1_score(y_new, y_pred)

    acc = accuracy_score(y_new, y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_new, y_pred)

    return acc, f_score, cm


def data_tests():
    dataprocessing.extract_random_10k("merged_dataset.csv")
    liar_data_filename = "test.tsv"

    liar_data = pd.read_csv(liar_data_filename, delimiter='\t')

    liar_data = liar_data.iloc[:, [1, 2]]

    liar_data.columns = ['type', 'content']

    sim_liar_accuracy = evaluate_simple_liar(liar_data)

    adv_liar_accuracy = evaluate_advanced_liar(liar_data)

    print("Simple model accuracy on liar test data:", sim_liar_accuracy)

    print("Advanced model accuracy on liar test data:", adv_liar_accuracy)

    data_filename = "10k_merged_dataset.csv"

    FakeNews_data = pd.read_csv(data_filename)

    sim_accuracy = evaluate_simple_merged(FakeNews_data)

    adv_accuracy = evaluate_advanced_merged(FakeNews_data)

    print("Advanced model accuracy on random subset of 995k data:", sim_accuracy)

    print("Advanced model accuracy on random subset of 995k data:", adv_accuracy)
data_tests()
