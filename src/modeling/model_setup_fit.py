import sys
import os
import psycopg2
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

"""Necessary database credentials"""
# should be dar prod
global HOST_DAR, USER_DAR, PASSWORD_DAR, DB_DAR
HOST_DAR = os.environ.get("HOST_DAR")
USER_DAR = os.environ.get("USER_DAR")
PASSWORD_DAR = os.environ.get("PASSWORD_DAR")
DB_DAR = os.environ.get("DB_DAR")

global HOST_FORKED, USER_FORKED, PASSWORD_FORKED, DB_FORKED
HOST_FORKED = os.environ.get("HOST_FORKED")
USER_FORKED = os.environ.get("USER_FORKED")
PASSWORD_FORKED = os.environ.get("PASSWORD_FORKED")
DB_FORKED = os.environ.get("DB_FORKED")


def get_table_from_db(query_src, type, HOST, USER, PASSWORD, DB):
    """Creates a pandas dataframe from a query to a Postgres database.
    Input Attributes: all are mandatory
        * **query_src**: input the name of the sql file as a string, e.g. *'get_raw_data.sql'*, OR the query itself as a string.
        * **type**: what form the query is in, 'string' or 'file'
        * **HOST,USER,PASSWORD,DB**: strings of your Postgres database credentials
    """
    cwd = os.getcwd()
    os.chdir(cwd+'/sql')

    print("Querying data from DB connection")
    if type == 'file':
        queryfile = open(query_src, 'r')
        query = queryfile.read()
        queryfile.close()
    else:
        query = query_src

    success = None
    print ("Trying to establish initial connection to the server")
    while success is None:
        conn = psycopg2.connect(host=HOST, user=USER, password=PASSWORD, dbname=DB)
        cur = conn.cursor()
        try:
            cur.execute(query)
            print("Success!")
            success = "true"
            break
        except psycopg2.DatabaseError:
            print('Check sql query if able to execute or check your db connection credentials.')
            pass

    names = [x[0] for x in cur.description]
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=names)

    if success is not None:
        conn.close()

    print("Finished querying data")
    os.chdir(cwd)
    return df


def drop_column_containing(data, col):
    """Deletes the column(s) of a dataframe that start with the string 'col'. Not unique to modeling so putting it outside the classes
    """
    col_list = data.columns.tolist()

    for c in col_list:
        if c.startswith(col) and c != 'postal_ak':
            data = data.drop(c, axis=1)
            print "Dropped: " + c

    return data


def get_clean_y_query(y, query_file):
    """Returns a query in the form of the string to obtain a specific variable y using the query in *query_file* (e.g.: *'get_yvar_onyx.sql'*)"""
    cwd = os.getcwd()
    os.chdir(cwd+'/sql')

    f = open(query_file, 'r')
    query = f.read()

    if (y != 'purpose_connect_category' and y != 'num_lines'):
        query = query.replace('yvar', y)
    if y == 'num_lines':
        query = query.replace('sr.yvar', 'li.' + y)

    os.chdir(cwd)
    return query


class Model(object):
    """Class to initialize a model to fit to given training data and y variable, and fit a model using GridSearch cross validation.
    Input Attributes: all are mandatory except for **p**:
        * **training_data**: the training dataset without the (clean) y variable
        * **yvar**: column name of y variable to predict
        * **model**: the type of model to fit. Options are one of: *['logisticregression', 'decisiontree', 'randomforest', 'gradientboosting']*
        * **classification_type**: *'classification'* or *'regression'*
        * **imputer_strategy**: how to handle missing values -- must be an array. Example 1: *imputer_strategy = ['mean']*. Example 2: *imputer_strategy = ['mean', 'median', 'most_frequent']*
        * **p**: proportion of the training dataset to set for training; number between 0 and 1 (optional, default = 0.8)
    """

    def __init__(self, training_data, yvar, model, classification_type, imputer_strategy, **kwargs):
        self.training_data = training_data
        self.yvar = yvar
        self.model = model
        self.classification_type = classification_type
        self.imputer_strategy = imputer_strategy
        self.p = kwargs.get('p', .8)
        # y variable (actual column, not name)
        self.y = pd.Series()
        # label encoder used for y variable
        self.le = None
        # Parameters, grid and best model from GridSearch
        self.pipe = None
        self.params = None
        self.grid = None
        self.bestimputer = None
        self.bestmodel = None

    def merge_y_variable(self, queryfunc, *args, **kwargs):
        """Runs the query for the y variable (given by ``get_clean_y_query(y,..)``) and adds it as a column to the training set by joining via *frn_adjusted*"""
        ydata = queryfunc(*args, **kwargs)
        self.training_data = pd.merge(self.training_data, ydata, on='frn_adjusted', how='inner')
        self.training_data = self.training_data.drop_duplicates()
        return self

    def label_encode(self):
        """Returns the y class variable in the format ``sklearn`` needs to run the model, as well as the ``LabelEncoder`` used"""
        le = preprocessing.LabelEncoder()
        le.fit(self.y)
        return le.transform(self.y), le

    def training_setup(self):
        """Prepares the feature set and y variable for ``sklearn`` modeling"""
        if self.yvar != 'fiber_binary':
            yquery = get_clean_y_query(self.yvar, 'get_yvar_dar_prod.sql')
        else:
            yquery = get_clean_y_query('connect_category', 'get_yvar_dar_prod.sql')

        # rename pristine/dirty y variable column in training data if it exists, attach clean y variable
        if self.yvar in self.training_data.columns:
            self.training_data.columns = self.training_data.columns.str.replace(self.yvar, self.yvar+'_pre')
        if "num_lines_pre" in self.training_data.columns.tolist():
            self.training_data["num_lines_pre"] = [0 if ele < 0 else ele for ele in self.training_data["num_lines_pre"]]

        self.merge_y_variable(get_table_from_db, yquery, 'string', HOST_DAR, USER_DAR, PASSWORD_DAR, DB_DAR)

        if self.yvar == 'num_lines':
            if self.classification_type == 'regression':
                self.training_data["num_lines"] = [0 if ele < 0 else ele for ele in self.training_data["num_lines"]]
            else:
                self.training_data["num_lines"] = [0 if ele > 1 else 1 for ele in self.training_data["num_lines"]]

        if self.yvar == 'fiber_binary':
            self.training_data["fiber_binary"] = [1 if ("Fiber" in ele or "ISP" in ele) else 0 for ele in self.training_data["connect_category"]]
            self.y = self.training_data["fiber_binary"]
            self.training_data = self.training_data.drop(['frn_adjusted', 'connect_category', 'fiber_binary'], axis=1)
        else:
            self.y = self.training_data[self.yvar]
            self.training_data = self.training_data.drop(['frn_adjusted', self.yvar], axis=1)

        if self.classification_type == 'classification' and self.yvar != 'fiber_binary' and self.yvar != 'num_lines':
            unique, counts = np.unique(self.y, return_counts=True)
            print "Category counts for " + self.yvar + "(pre-encoding):"
            print np.asarray((unique, counts)).T
            self.y, self.le = self.label_encode()

        return self

    def get_train_test_split(self):
        """Splits the training features X and y variable according to the proportion p. Returns 4 elements (X and y for training & testing)"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.training_data, self.y, train_size=self.p, random_state=7)
        return self

    def build_pipe(self, **kwargs):
        """Primary method: prepares the training and test sets and model pipeline to input into GridSearch. Optional input attributes (keyword arguments) correspond to model parameter ARRAYS to test via GridSearch
        and depend on the model chosen. Please refer to the ``sklearn`` docs for your desired model type, e.g. ``sklearn.ensemble.RandomForestClassifier``.
            * Example for *model = 'randomforest'*: ``build_pipe(n_estimators = [50,100,300], class_weight = ['balanced'])``
        """
        self.training_setup().get_train_test_split()

        if self.model == 'logisticregression':
            self.pipe = Pipeline([("imputer", Imputer()), ("estimator", LogisticRegression(random_state=7))])

        if self.model == 'decisiontree':
            if self.classification_type == 'classification':
                self.pipe = Pipeline([("imputer", Imputer()), ("estimator", DecisionTreeClassifier(random_state=7))])
            else:
                self.pipe = Pipeline([("imputer", Imputer()), ("estimator", DecisionTreeRegressor(random_state=7))])

        if self.model == 'randomforest':
            if self.classification_type == 'classification':
                self.pipe = Pipeline([("imputer", Imputer()), ("estimator", RandomForestClassifier(random_state=7))])
            else:
                self.pipe = Pipeline([("imputer", Imputer()), ("estimator", RandomForestRegressor(random_state=7))])

        if self.model == 'gradientboosting':
            if self.classification_type == 'classification':
                self.pipe = Pipeline([("imputer", Imputer()), ("estimator", GradientBoostingClassifier(random_state=7))])
            else:
                self.pipe = Pipeline([("imputer", Imputer()), ("estimator", GradientBoostingRegressor(random_state=7))])

        self.params = dict(('estimator__' + name, kwargs[name]) for name in kwargs)
        self.params.update({'imputer__strategy': self.imputer_strategy})

        return self

    def print_score(self):
        """
        prints accuracy and confusion matrices for a given classification model that indicate performance.
        If regression model, just prints the MSE
        """
        pd.set_option('expand_frame_repr', False)
        for sets in ['train', 'test']:
            X = getattr(self, 'X_' + sets)
            y = getattr(self, 'y_' + sets)
            print "Result for: " + sets
            y_pred = self.grid.predict(X)

            if self.classification_type == 'classification':
                print accuracy_score(y, y_pred)
                if self.yvar != 'fiber_binary' and self.yvar != 'num_lines':
                    cm = pd.crosstab(self.le.inverse_transform(y), self.le.inverse_transform(y_pred), rownames=['True'], colnames=['Predicted'], margins=True)
                else:
                    cm = pd.crosstab(y, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
                print cm
                sys.stdout.flush()
            else:
                print mean_squared_error(y, y_pred)
                sys.stdout.flush()

    def print_feature_importances(self):
        """
        prints feature importances for a given model
        """
        tuples = list(zip(self.X_train.columns, self.bestmodel.feature_importances_))
        feature_importances = pd.DataFrame(tuples, columns=['Feature', 'Importance'])
        print feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)
        sys.stdout.flush()

    def get_feature_importances(self):
        """
        gets feature importances for a given model and outputs a dataframe
        """
        tuples = list(zip(self.X_train.columns, self.bestmodel.feature_importances_))
        feature_importances = pd.DataFrame(tuples, columns=['Feature', 'Importance'])
        return feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)
        sys.stdout.flush()

    def randomized_fit(self, **kwargs):
        """To assist in narrowing down the hyperparameter settings. Fits the model specified in ''build_pipe()'' via RandomizedSearchCV.
            * **cv**: integer number of folds to use for cross validation (optional, default = 3)
            * **scoring**: metric to be optimized for (optional, default = 'accuracy')
            * **verbose**: integer indicating how much output you want to see from GridSearch (optional, default = 3)
            * **n_jobs**: integer indicating how many parallel jobs to run (optional, default = 1)
            * **n_iter**: integer indicating how many samples to take of the parameter settings. Tradeoff is runtime vs quality of solution (optional, default = 100)
            * Example: ``randomized_fit(cv=4, verbose=6, n_jobs=6)``
        """
        # Use the random grid to search for best hyperparameters
        # Random search of parameters, using 3 fold cross validation,
        # search across 100 different combinations, and use all available cores

        cv = kwargs.get('cv', 3)
        if self.classification_type == 'classification':
            scoring = kwargs.get('scoring', 'accuracy')
        else:
            scoring = kwargs.get('scoring', 'neg_mean_squared_error')
        verbose = kwargs.get('verbose', 3)
        n_jobs = kwargs.get('n_jobs', 1)
        n_iter = kwargs.get('n_iter', 100)

        random_grid = RandomizedSearchCV(self.pipe, self.params, cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs, n_iter=n_iter)
        # Fit the random search model
        random_grid.fit(self.X_train, self.y_train)
        print "Finished fit for all iterations."

        print "BEST PARAMETERS: " + str(random_grid.best_params_)

    def fit(self, **kwargs):
        """Primary method: fit the model specified in ``build_pipe()`` via GridSearch. Input attributes correspond to the inputs to GridSearchCV (http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
        Note only the 3 listed below are applicable (rest are set implicitly):
            * **cv**: integer number of folds to use for cross validation (optional, default = 3)
            * **scoring**: metric to be optimized for (optional, default = 'accuracy')
            * **verbose**: integer indicating how much output you want to see from GridSearch (optional, default = 3)
            * **n_jobs**: integer indicating how many parallel jobs to run (optional, default = 1)
            * Example: ``fit(cv=4,verbose=6,n_jobs=6)``
        """
        cv = kwargs.get('cv', 3)
        if self.classification_type == 'classification':
            scoring = kwargs.get('scoring', 'accuracy')
        else:
            scoring = kwargs.get('scoring', 'neg_mean_squared_error')
        verbose = kwargs.get('verbose', 3)
        n_jobs = kwargs.get('n_jobs', 1)

        grid = GridSearchCV(self.pipe, self.params, cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs)
        grid.fit(self.X_train, self.y_train)

        self.grid = grid
        self.bestimputer = grid.best_estimator_.steps[0][1]
        self.bestmodel = grid.best_estimator_.steps[1][1]

        self.print_feature_importances()
        self.print_score()
