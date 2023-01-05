# Disaster Response Project

## Table of Contents
 * [Neceesary packages](#Necessary-packages)
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [How to interact with this project](#how-to-interact-with-this-project)
 * [Results](#Results)
 * [Summary](#Summary)
 * [Acknowledgements](#Acknowledgements)



### Necessary packages:

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


### Project Motivation:

The goal of this project is to create a machine learning pipeline to categorize disaster messages into categories.

### File Descriptions:

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

### How to interact with this project
- clone the project
- To run the ETL pipeline that will store the final data in a database, run `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db`
- To run the ML pipeline, run: `python models/train_classifier.py data/disaster_response.db models/model.pkl`
- to start the web app, navigate into the app-folder: '<yourFilepath>/Project-2/app/' and type `python run.py`.
- Go to http://0.0.0.0:3001/
- improve the web-app with better visualizations. Feel free to optimize the machine-learning model, use a different one etc. :)

### Results: 

- an ETL pipeline and a machine learning model build with Pipelines
- a Flask web-app

### Summary: 

Data had roughtly 26000 messages. These messages were categorized into 36 categories.  

These were the messages with the most frequent categories:

- related: 20316
- aid_related: 10878
- weather_related: 7304
- direct_report: 5081
- request: 4480

A machine learning pipeline was build, 'message' being the input column and the 36 categories being the target columns.

The machine-learning model is saved as a .pkl-file.

In the web-app, this .pkl file is the "brain" that classifies the input sentence in the backend.


### Acknowledgements:
- udacity for giving feedback on the code
- [stephanieirvine](https://github.com/stephanieirvine/Udacity-Data-Scientist-Nanodegree/blob/main/Project%202/ML%20Pipeline%20Preparation.ipynb) as I took one function from her.