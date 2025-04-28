import json
import plotly
import pandas as pd
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

# Download punkt if not already present
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Just in case it's not available

app = Flask(__name__)

# Custom tokenize function
def tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize
    tokens = word_tokenize(text)

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]

    return clean_tokens

# Load data
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, '..', 'data', 'DisasterResponse.db')
engine = create_engine(f'sqlite:///{db_path}')
df = pd.read_sql_table('DisasterResponse', engine)

# Load model
model = joblib.load("models/classifier.pkl")

# Index route
@app.route('/')
@app.route('/index')
def index():
    # Genre distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Top 10 categories
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False).head(10)
    category_names = category_counts.index.tolist()

    # Graphs
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category", 'tickangle': -45}
            }
        }
    ]

    # Encode plotly graphs
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Go route: handles classification
@app.route('/go')
def go():
    query = request.args.get('query', '') 

    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
