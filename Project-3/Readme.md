# Udacity Data Scientist Project 3: Recommendation Engines

## Table of Contents
 * [Neceesary packages](#Necessary-packages)
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [How to interact with this project](#how-to-interact-with-this-project)
 * [Results](#results)
 * [Acknowledgements](#Acknowledgements)



### Necessary packages
- pandas 
- numpy
- matplotlib.pyplot
- project_tests
- pickle

### Project Motivation

The goal of this project is to build a recommendation system on data from the IBM Watson Studio platform. 


### File Descriptions

- Recommendations_with_IBM.ipynb: Jupyter Notebook the analyis
- Recommendations_with_IBM.html: Jupyter Notebook as .html
- project_tests.py: testing document, used to check results in Jupyter Notebook
- top_5.p: top 5 articles
- top_10.p: top 10 articles
- top_20.p: top 20 articles
- user_item_matrix.p: user item matrix
- data/articles_community.csv: raw data
- data/user-item-interaction.csv: raw data


### How to interact with this project
- clone the project `git clone https://github.com/TobiasGroeschner/udacity-data-scientist.git` and play around with the Notebook.

### Results
#### Part I : Exploratory Data Analysis

- 50% of individuals have 3 or fewer interactions.
- The total number of user-article interactions in the dataset is 45993.
- The maximum number of user-article interactions by any 1 user is 364.
- The most viewed article in the dataset was viewed 937 times.
- The article_id of the most viewed article is 1429.0'.
- The number of unique articles that have at least 1 rating 714.
- The number of unique users in the dataset is 5148.
- The number of unique articles on the IBM platform.

#### Part II: Rank-Based Recommendations
- a function that returns top articles
- a functions that returns top articles ids

####  Part III: User-User Based Collaborative Filtering
- a function that creates a user item matrix
- a function that finds similiar users
- a function that returns a list of article names associated with the list of article ids 
- a function that provides a list of the article_ids and article titles that have been seen by a user
- a function that finds articles the user hasn't seen before and provides them as recommendations
- a function:  For each user - finds articles the user hasn't seen before and provides them as recommendations

#### Part IV: Content Based Recommendations (EXTRA - NOT REQUIRED): not done

#### Part V: Matrix Factorization

- Singular Value Decomposition on the user-item matrix
- created create_user_item_matrix with train and test data
- analysis of accuracy

### Acknowledgements
- udacity for giving feedback on the code
- https://github.com/nikextens/Data_Scientist_Recommendations_with_IBM - some code-parts were taken and adjusted
- https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.svd.html
- https://www.geeksforgeeks.org/python-set-difference/
