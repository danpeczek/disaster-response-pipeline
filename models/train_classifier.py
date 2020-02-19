import sys
import os
from sqlalchemy import create_engine
import pandas as pd
import re
import pickle

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    database_string = "sqlite:///" + database_filepath
    engine = create_engine(database_string)
    db_name = database_filepath.split(os.path.sep)[-1]
    db_name = db_name.split('.')[0]
    df = pd.read_sql('SELECT * FROM '+db_name, con=engine)
    X = df['message'].values
    Y = df.iloc[:, 3:].values
    category_names = df.iloc[:, 3:].columns
    return X, Y, category_names


def tokenize(text):

    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if stopwords.words('english')]
    tokens = [PorterStemmer().stem(w) for w in tokens]
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])

    parameters = {
        'vectorizer__ngram_range': [(1, 2), (2, 3), (3, 4)],
        'vectorizer__max_df': [1.0],
        'vectorizer__max_features': [25000, 30000],
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [31, 51, 71]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=100, cv=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    print('*** Best Parameters found during cross validation')
    print(model.best_params_)

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
