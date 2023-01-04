# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import re

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath:str):
    
    """    
    Loads the dataset

    Parameters:
    database_filepath: absolute filepatch to the database 
    
    Returns:
    X: np.array --> feature columns
    Y: pd.DataFrame --> target columns
    """
    
    #disaster_response.db is the name of the file
    engine = create_engine('sqlite:///'+ database_filepath)
    #DisasterResponse is the table name
    df =  pd.read_sql_table("DisasterResponse.db", con=engine)

    #define feature columns
    X = df['message'].values

    #define the target columns
    Y = df[df.columns[4::]]
    
    return X,Y, 


def tokenize(text)-> list:
    """
    Tokenizes and lemmatizes the text

    Parameters:
    text: Text to be tokenized
    
    Returns:
    clean_tokens: a list of cleaned tokens
    """

    #split text into tokens
    tokens = word_tokenize(text)
    #create WordNetLemmatizer instance
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    """    
    Builds the pipeline

    Parameters:
    None
    
    Returns:
    pipeline: a pipeline object
    """

    #create a pipeline object
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test)->None: 
    """
    Predicts the model and prints out a classification report for each column.

    Parameters:
    model: Pipeline object 
    X_test: test data of feature columns
    Y_test: test data of target column
    
    Returns:
    None
    """

    #predict the model
    y_pred = model.predict(X_test)

    #iterate through each index & and column and build a classifcation report
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))
    
    
def save_model(model, model_filepath):
    """
    Exports the final model as a pickle file
    
    Parameters:
    model: Pipeline object 
    model_filepath: filepath where to save the model

    Returns:
    None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():

    database_filepath = "/home/tobias_groeschner/projects/DataScience/Project-2/disaster_response.db"
    model_filepath = "/home/tobias_groeschner/projects/DataScience/Project-2/model.pkl"
    sys.argv.append(database_filepath)
    sys.argv.append(model_filepath)
    
    
    if len(sys.argv) == 3:
    
       print('Loading data...\n    DATABASE: {}'.format(database_filepath))
       X, Y = load_data(database_filepath)
       X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
       
       print('Building model...')
       model = build_model()

       print('Training model...')
       model.fit(X_train, Y_train)
       
       print('Evaluating model...')
       evaluate_model(model, X_test, Y_test)

       print('Saving model...\n    MODEL: {}'.format(model_filepath))
       save_model(model, model_filepath)
#
       print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()