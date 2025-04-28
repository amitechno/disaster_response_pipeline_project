import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Ensure NLTK corpora are downloaded (run once)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))
# Use RegexpTokenizer to avoid NLTK punkt dependency
tokenizer = RegexpTokenizer(r"\w+")


def load_data():
    """
    Load data from the SQLite database and split into features and labels.

    Returns:
        X (pd.Series): Messages.
        Y (pd.DataFrame): Category labels.
        category_names (list): List of category names.
    """
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y, Y.columns.tolist()


def tokenize(text):
    """
    Normalize, tokenize using RegexpTokenizer, remove stop words, and lemmatize the input text.
    """
    # Lowercase
    text = text.lower()
    # Tokenize to words (alphanumeric)
    tokens = tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).strip() 
                    for tok in tokens if tok not in STOP_WORDS]
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline and wrap it in GridSearchCV.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1, error_score='raise')
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance and print classification report for each category.
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        print(f"Category: {col}\n", classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model):
    """
    Save the trained model as a pickle file.
    """
    to_save = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
    joblib.dump(to_save, 'models/classifier.pkl')


def main():
    print('Loading data from data/DisasterResponse.db...')
    X, Y, category_names = load_data()

    print('Splitting data into train and test sets...')
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print('Building model pipeline...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model to models/classifier.pkl...')
    save_model(model)

    print('Model training and saving complete.')

if __name__ == '__main__':
    main()
