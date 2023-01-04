import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify #render_template: what html file should show up based on a web address
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
database_filepath = "/home/tobias_groeschner/projects/DataScience/Project-2/disaster_response.db"
engine = create_engine('sqlite:///' + database_filepath)
df = pd.read_sql_table('DisasterResponse.db', engine)

# load model
model_filepath = "/home/tobias_groeschner/projects/DataScience/Project-2/model.pkl"
model = joblib.load(model_filepath)


# index webpage displays cool visuals and receives user input text for model
@app.route('/') 
@app.route('/index') #if the user goes to the homepage, the the index.html file is rendered
def index():
    
    # extract data needed for visuals
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
   
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Count of genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(df["request"].value_counts().index),
                    y=df["request"].value_counts()
                )
            ],

            'layout': { 
                'title': 'Count of request',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "0 = False, 1 = True"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(df["aid_related"].value_counts().index),
                    y=df["request"].value_counts()
                )
            ],

            'layout': { 
                'title': 'Count of aid_related',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "0 = False, 1 = True"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()