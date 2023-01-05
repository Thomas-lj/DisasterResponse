import sys
import os
import re
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
<<<<<<< HEAD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sqlalchemy import create_engine

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
=======
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sqlalchemy import create_engine
>>>>>>> 3461bcf41f583ceeecc3d96a9c7a00c64f2e8a35
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def load_data(database_filepath):
<<<<<<< HEAD
    """
    loads the data saved in a db. 
    Input:
        database_filepath: str, ends with .db
    ------------
    Returns:
        X: pd.DataFrame features
        Y: ML labels
        Y.columns: ML label names
    """
    db_path = 'sqlite:///' + os.getcwd() + '/' + database_filepath
    engine = create_engine(db_path)
    df = pd.read_sql_table('disaster_table', con=engine)
    df = df.drop(columns=['id', 'original', 'genre'])
    X = df['message']
    Y = df.loc[:, df.columns != 'message']
    return X, Y, list(Y.columns)

def tokenize(text):
    """
    This functions tokenizes a str text.
    Input:
        text, str
    ------------
    Returns:
        tokens, list
    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
=======
    db_path = 'sqlite:///' + os.getcwd() + '/' + database_filepath
    engine = create_engine(db_path)
    df = pd.read_sql('SELECT * FROM disaster_table', engine)
    df = df.drop(columns=['id', 'original', 'genre'])
    X = df['message']
    Y = df.loc[:, df.columns != 'message']
    return X, Y, Y.columns

def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
>>>>>>> 3461bcf41f583ceeecc3d96a9c7a00c64f2e8a35
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
<<<<<<< HEAD
    tokenized = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokenized if word not in stop_words]
=======
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
>>>>>>> 3461bcf41f583ceeecc3d96a9c7a00c64f2e8a35
    return tokens


def build_model():
<<<<<<< HEAD
    """
    This functions builds the data processing pipeline.
    ------------
    Returns:
        pipeline, Pipeline instance.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [10, 25],
        'clf__estimator__min_samples_split': [2, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate the model with its X_test and ground truth y_test labels.
    Input:
        model, sklearn Pipeline instance
        X_test, df
        y_test, labels
        category_names, y_test column labels
    ------------
    prints the classification summary of the model.
    """

    y_pred = model.predict(X_test)
    class_rep = classification_report(y_test.values, y_pred, target_names=category_names)
    print("Classification report:\n", class_rep)

def save_model(model, model_filepath):
    """
    Save the trained model in model_filepath (str).
    """
=======
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=category_names)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", category_names)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)

def save_model(model, model_filepath):
>>>>>>> 3461bcf41f583ceeecc3d96a9c7a00c64f2e8a35
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
<<<<<<< HEAD
=======
        tokenize(X.iloc[0].message)
>>>>>>> 3461bcf41f583ceeecc3d96a9c7a00c64f2e8a35
        print('Building model...')
        model = build_model()
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()