# Udacity Data Scientist Project 3: Recommendation Engines

## Table of Contents
 * [Neceesary packages](#Necessary-packages)
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [How to interact with this project](#how-to-interact-with-this-project)
 * [Results](#results)
 * [Acknowledgements](#Acknowledgements)



### Necessary packages

see poetry.lock file

### Project Motivation

The goal of this project is to analyze demographic data for customers of a mail-order sales company in Germany. We will compare the demographic information of the customers with the general population. This project uses unsupervised learning for customer segmentation. Furthmore, a model will be developed predicting individuals most likely to become customers of the company. The data was provided by Bertelsmann Arvato Analytics.


### File Descriptions

- `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
- `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
- `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
- `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

### How to interact with this project
- clone the project `git clone https://github.com/TobiasGroeschner/udacity-data-scientist.git` and play around with the Notebook.

### Results


#### Data Prerocessing
 - discard columns with above 0.4 NaN values
 - factorize object columns so that machine learning algorithms can handle data
 - used (SimpleImputer)[https://scikit-learn.org/stable/modules/impute.html] to fill remaining NaN values
 - used (StandardScaler)[https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html] to normalize the data


#### Customer segmentation report
 - used (Principal Component Analysis)[https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html] to reduce data
 - used (Clustering)[https://towardsdatascience.com/how-many-clusters-6b3f220f0ef5] to cluster & compare the population vs. the customer data
 - clusters of most influence to become a customer was based on the lifestage of the person.

#### Machine learning

 - Used GridSearch to find optimal hyperaramters
 - Best score we achieved: 0.69% above ROC curve on testing data


### Acknowledgements

- udacity for providing this project
- the mail order company for providing the data
- Aléaume as I took some code from him: https://github.com/Aleaume/Udacity_DataSc_P4/tree/main

### Sources

- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://scikit-learn.org/stable/modules/impute.html
- https://builtin.com/data-science/step-step-explanation-principal-component-analysis
- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- https://towardsdatascience.com/how-many-clusters-6b3f220f0ef5
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score

