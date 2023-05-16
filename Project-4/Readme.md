# Udacity Data Scientist Project 3: Recommendation Engines

## Table of Contents
 * [Neceesary packages](#Necessary-packages)
 * [Project Motivation](#project-motivation)
 * [File Descriptions](#file-descriptions)
 * [How to interact with this project](#how-to-interact-with-this-project)
 * [Results](#results)
 * [Acknowledgements](#Acknowledgements)



### Necessary packages

see ../poetry.lock file or requirements.txt 

### Project Motivation

The goal of this project is to analyze demographic data for customers of a mail-order sales company in Germany. We will compare the demographic information of the customers with the general population. This project uses unsupervised learning for customer segmentation. Furthmore, a model was developed predicting individuals most likely to become customers of the company. The data was provided by Bertelsmann Arvato Analytics.


### File Descriptions

- `requirements.txt`:  a list of all installed Python modules to complete the Data Science nanodegree with their versions.
- `images/*`: images used for medium report
- `DIAS Attributes - Values 2017.xlsx`: explanatory excel
- `DIAS Information Levels - Attributes 2017.xlsx`: explanatory excel
- `Arvato Project Workbook.ipynb`: project notebook
- `functions.py`: support functions
- `process_data.py`: functions for processing data

The remaining files are not uploaded due to rules of udacity.

### How to interact with this project
- clone the project `git clone https://github.com/TobiasGroeschner/udacity-data-scientist.git` and play around with the Notebook.

### Results

Read the a [blog post](https://medium.com/@tobias.groeschner/udacity-data-scientist-project-4-customer-segmentation-report-for-arvato-financial-services-eb3d9095ee6d) on medium.

#### Data Prerocessing
 - discard columns with above 0.4 NaN values
 - factorize object columns so that machine learning algorithms can handle data
 - used [SimpleImputer](https://scikit-learn.org/stable/modules/impute.html) to fill remaining NaN values
 - used [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to normalize the data


#### Customer segmentation report
 - used [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to reduce data
 - used [Clustering](https://towardsdatascience.com/how-many-clusters-6b3f220f0ef5) to cluster & compare the population vs. the customer data
 - clusters of most influence to become a customer was based on the lifestage of the person.

#### Machine learning

 - Used GridSearch to find optimal hyperaramters
 - Best score we achieved: 0.69% above ROC curve on testing data


### Acknowledgements

- udacity for providing this project
- the mail order company for providing the data
- Aléaume as his code helped me to complete this project: https://github.com/Aleaume/Udacity_DataSc_P4/tree/main

### Sources

Data Processing
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- https://scikit-learn.org/stable/modules/impute.html

Customer segmentation
- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- https://builtin.com/data-science/step-step-explanation-principal-component-analysis


Clustering
- https://towardsdatascience.com/how-many-clusters-6b3f220f0ef5
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

Machine Learning
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

