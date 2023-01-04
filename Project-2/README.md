Necessary packages:

- import pandas as pd
- import matplotlib.pyplot as plt

- import seaborn as sb
- import sys
- from sqlalchemy import create_engine

- from nltk.tokenize import word_tokenize
- from nltk.stem import WordNetLemmatize

- from sklearn.metrics import confusion_matrix
- from sklearn.model_selection import train_test_split
- from sklearn.pipeline import Pipeline
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
- from sklearn.multioutput import MultiOutputClassifier
- from sklearn.metrics import classification_report
- from sklearn.model_selection import GridSearchCV


Project Motivation:The goal of this project is to create a machine learning pipeline to categorize disaster messages into categories.

File Descriptions:
- ETL Pipeline Preparation.ipynb
    - Loads the *disaster_messages.csv* and *disaster_categories.csv* datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database
- ML Pipeline Preparation.ipynb
    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file
- disaster_response.db: necessary data
- app/run.py: python file to run to start a web abb
- data/disaster_cagegories.csv: raw data
- data/disaster_messages.csv: raw data
- data/process_data.py: ETL Pipeline Preparation.ipynb in a python file
- models/train_classifier.py: ML Pipeline Preparation.ipynb in a python file

How to interact with this project?
- clone the project
- to start the web app, navigate into the app-folder: '<yourFilepath>/DataScience/Project-2/app/' and type 'python run.py'. 
- improve the web-app with better visualizations. Feel free to optimize the machine-learning model, use a different one etc. :)

Results: 

- an ETL pipeline and a machine learning model build with Pipelines
- a Flask web-app

Acknowledgements:
- udacity for giving feedback on the code
- [stephanieirvine](https://github.com/stephanieirvine/Udacity-Data-Scientist-Nanodegree/blob/main/Project%202/ML%20Pipeline%20Preparation.ipynb) as I took one function from her.