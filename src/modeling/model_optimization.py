from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from model_setup_fit import *
import numpy as np
from cPickle import dump, load, HIGHEST_PROTOCOL
import json
from psycopg2.extensions import AsIs
from psycopg2.extras import Json, DictCursor
import copy
from datetime import datetime

"""ML Staging credentials"""
global HOST_FORKED, USER_FORKED, PASSWORD_FORKED, DB_FORKED
HOST_FORKED = os.environ.get("HOST_FORKED")
USER_FORKED = os.environ.get("USER_FORKED")
PASSWORD_FORKED = os.environ.get("PASSWORD_FORKED")
DB_FORKED = os.environ.get("DB_FORKED")


class ModelOptimizer(object):
    """Class to optimize a ``model_setup_fit.Model`` model via feature elimination with a user-specified strategy and store the final model, feature list and results.
    Input Attributes:
        * **model_fit_obj**: An instance of the ``model_setup_fit.Model`` class that has been fit (i.e. has a ``bestmodel`` attribute other than *None*), mandatory.
        * **strategy**: Feature elimination testing strategy, mandatory. Options are one of: *['importance','manual','both']*
            * *'importance'*:  For each feature importance value *v* of each variable in ascending order, test each subset of features with importance >= *v*, starting with all features and ending with the subset with the most important feature.
            * *'manual'*: Eliminate the features specified in **drop_features** only
            * *'both'*: Eliminate the features specified in **drop_features** AND those with an importance value >= **threshold**. Note, must supply a values for both **drop_features** and **threshold** if using this strategy.
        * **drop_features**: A list of column names of feature(s) to test elimination of, optional.
        * **threshold**: If given, we only use the subset of features with an importance value >= **threshold** for all strategies.
    """

    def __init__(self, model_fit_obj, strategy, **kwargs):
        self.strategy = strategy
        if strategy == 'manual' and ('drop_features' not in kwargs):
            print "Error: need to provide at least one feature to drop"
            return
        if strategy == 'both' and ('drop_features' not in kwargs or 'threshold' not in kwargs):
            print "Error: need to provide a an importance threshold and at least one feature to drop"
            return
        self.drop_features = kwargs.get('drop_features', None)
        self.threshold = kwargs.get('threshold', None)
        self.features = model_fit_obj.X_train.columns
        self.yvar = model_fit_obj.yvar
        self.model = model_fit_obj.model
        self.classification_type = model_fit_obj.classification_type
        self.le = model_fit_obj.le
        self.bestmodel = copy.deepcopy(model_fit_obj.bestmodel)
        self.bestimputer = model_fit_obj.bestimputer
        self.X_train = model_fit_obj.X_train
        self.X_test = model_fit_obj.X_test
        self.y_train = model_fit_obj.y_train
        self.y_test = model_fit_obj.y_test

    def getfeatures(self):
        return self.features

    def eliminate_importance(self):
        """
        Eliminates features by variable importance. If given a threshold, removes all variables an importance value less than the threshold.
        """
        imputer = self.bestimputer
        best_model = self.bestmodel
        starting_X = self.X_train[self.features.tolist()]
        starting_Xtest = self.X_test[self.features.tolist()]
        Xtrain = imputer.fit_transform(starting_X)
        Xtrain = pd.DataFrame(Xtrain, index=starting_X.index, columns=starting_X.columns)
        Xtest = imputer.fit_transform(starting_Xtest)
        Xtest = pd.DataFrame(Xtest, index=starting_Xtest.index, columns=starting_Xtest.columns)

        if self.threshold is None:
            # Fit model using each importance as a threshold
            thresholds = sorted(self.bestmodel.feature_importances_)
            thresholds = [t for t in thresholds if t > 0.00000000]
        else:
            thresholds = [self.threshold]

        accuracy_list = []
        for thresh in thresholds:
        ## for thresh in thresholds[:200:5]: ##commenting out threshold iteration used in gradient boosting model to speed things up
            # select features using threshold
            selection = SelectFromModel(best_model, threshold=thresh)
            select_X_train = selection.fit_transform(Xtrain,self.y_train)

            # train model
            selection_model = self.bestmodel
            selection_model.fit(select_X_train, self.y_train)
            # eval model
            select_X_test = selection.transform(Xtest)
            y_pred = selection_model.predict(select_X_test)

            if self.classification_type == 'classification':
                accuracy = accuracy_score(self.y_test, y_pred)
                accuracy_list.append(accuracy)
                if self.yvar != 'fiber_binary':
                    precision, recall, f_score, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')
                else:
                    precision, recall, f_score, _ = precision_recall_fscore_support(self.y_test, y_pred, average='binary')
                print("Thresh=%.8f, n=%d, Accuracy: %.2f%%, Precision: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0, precision * 100.0))
                sys.stdout.flush()
            else:
                mse = mean_squared_error(self.y_test, y_pred)
                print("Thresh=%.8f, n=%d, MSE: %.2f" % (thresh, select_X_train.shape[1], mse))
                sys.stdout.flush()

            if thresh == self.threshold:
                self.features = starting_X.columns[selection.get_support()]
                self.bestmodel = selection_model
                self.X_train = select_X_train
                self.X_test = select_X_test
        print("Maximum Accuracy: ", str(max(accuracy_list)))
        sys.stdout.flush()

    def eliminate_manual(self):
        """
        Eliminates features manually if supplied in **drop_features**
        """
        imputer = self.bestimputer
        Xtrain = imputer.fit_transform(self.X_train)
        Xtrain = pd.DataFrame(Xtrain, index=self.X_train.index, columns=self.X_train.columns)
        Xtest = imputer.fit_transform(self.X_test)
        Xtest = pd.DataFrame(Xtest, index=self.X_test.index, columns=self.X_test.columns)

        select_X_train = Xtrain.drop(self.drop_features, axis=1)

        # train model
        selection_model = self.bestmodel
        selection_model.fit(select_X_train, self.y_train)
        # eval model
        select_X_test = Xtest.drop(self.drop_features, axis=1)
        y_pred = selection_model.predict(select_X_test)

        if self.classification_type == 'classification':
            accuracy = accuracy_score(self.y_test, y_pred)
            if self.yvar != 'fiber_binary':
                precision, recall, f_score, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')
            else:
                precision, recall, f_score, _ = precision_recall_fscore_support(self.y_test, y_pred, average='binary')
            print("n=%d, Accuracy: %.2f%%, Precision: %.2f%%" % (select_X_train.shape[1], accuracy * 100.0, precision * 100.0))
        else:
            mse = mean_squared_error(self.y_test, y_pred)
            print("n=%d, MSE: %.2f%%" % (select_X_train.shape[1], mse * 100.0))

        self.features = select_X_train.columns
        self.bestmodel = selection_model
        self.X_train = select_X_train
        self.X_test = select_X_test

    def optimize(self):
        """Primary method: performs feature elimination using the strategy supplied
        """
        if self.strategy == 'importance':
            self.eliminate_importance()
        elif self.strategy == 'manual':
            self.eliminate_manual()
        else:
            self.eliminate_manual()
            self.eliminate_importance()


def get_columns_containing(data, col):
    """Obtains the column(s) of a dataframe that start with the string 'col' and returns the list of column names
    """
    col_list = data.columns.tolist()
    cols = []
    for c in col_list:
        if c.startswith(col):
            cols.append(c)

    return cols


def generate_id(model_obj, **kwargs):
    """Helper function to obtain the model id for a model (refer to Wiki for reference)
    """
    with open('lookup/id_value_lookup.json', 'r') as fp:
        id_value_lookup = json.load(fp)
    id_array = []
    if model_obj is not None:
        id_array.append(str(id_value_lookup['model'][model_obj.model]))
        id_array.append(str(id_value_lookup['imputer_strategy'][model_obj.bestimputer.strategy]))
        id_array.append(str(id_value_lookup['yvar'][model_obj.yvar]))
    else:
        id_array.append(str(id_value_lookup['model'][kwargs['model']]))
        id_array.append(str(id_value_lookup['imputer_strategy']['none']))
        id_array.append(str(id_value_lookup['yvar'][kwargs['yvar']]))

    # increment the max id
    query = """select max(dig4) as max_4dig from (
    select model_id, model_id::integer % 10 as dig4
    from dm.ml_model_results_lookup
    where left(model_id,3) = '""" + ''.join(id_array) + """') t;"""

    max_data = get_table_from_db(query, 'string', HOST_FORKED, USER_FORKED, PASSWORD_FORKED, DB_FORKED)
    id_array.append(str(max_data['max_4dig'].item() + 1) if (not max_data['max_4dig'].item() is None) else '1')

    model_id = ''.join(id_array)

    return model_id


def build_scoresdict(model_obj, **kwargs):
    """Helper function to build a dictionary of model scores for the training and testing sets.
    Scores in the dictionary include accuracy, precision, recall and mean squared error.
    """
    best_imputer = model_obj.bestimputer
    best_model = model_obj.bestmodel
    if model_obj is not None:
        for sets in ['train', 'test']:
            X = getattr(model_obj, 'X_' + sets)
            # apply the best imputer strategy
            X = best_imputer.fit_transform(X)
            y = getattr(model_obj, 'y_' + sets)

            y_pred = best_model.predict(X)

            if model_obj.classification_type == 'classification':
                accuracy = accuracy_score(y, y_pred)
                if model_obj.yvar != 'fiber_binary':
                    precision, recall, f_score, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
                else:
                    precision, recall, f_score, _ = precision_recall_fscore_support(y, y_pred, average='binary')
            else:
                accuracy = precision = recall = None

            if sets == 'train':
                train_scores = {'accuracy_train': accuracy,
                                'precision_train': precision,
                                'recall_train': recall,
                                'mse_train': mean_squared_error(y, y_pred)}
            if sets == 'test':
                test_scores = {'accuracy_test': accuracy,
                               'precision_test': precision,
                               'recall_test': recall,
                               'mse_test': mean_squared_error(y, y_pred)}

        scoresdict = dict(train_scores, **test_scores)
    else:
        scoresdict = {'accuracy_train': kwargs.get('accuracy_train', None),
                      'precision_train': kwargs.get('precision_train', None),
                      'recall_train': kwargs.get('recall_train', None),
                      'mse_train': kwargs.get('mse_train', None),
                      'accuracy_test': kwargs.get('accuracy_test', None),
                      'precision_test': kwargs.get('precision_test', None),
                      'recall_test': kwargs.get('recall_test', None),
                      'mse_test': kwargs.get('mse_test', None)}

    return scoresdict


def build_resultsdict(model_obj, features, **kwargs):
    """Helper function to build a dictionary of the scores obtained in ``build_scoresdict()`` as well as other info to store in the results database table such as the ``model_id`` and model parameters
    """
    model_id = generate_id(model_obj, **kwargs)
    resultsdict = {'model_id': model_id,
    'y': model_obj.yvar if model_obj is not None else kwargs['yvar'],
    'nfeatures': len(features),
    'imputer': model_obj.bestimputer.strategy if model_obj is not None else 'none',
    'classifier': model_obj.model if model_obj is not None else 'narrative_R',
    'classifier_params': model_obj.bestmodel.get_params() if model_obj is not None else dict(narrative_R=None)}

    scoresdict = build_scoresdict(model_obj, **kwargs)
    resultsdict = dict(resultsdict, **scoresdict)

    return resultsdict


def output_results(model_obj, features, name, **kwargs):
    """Obtains all model performance information (and other relevant stats) and populates the table ``dm.ml_model_results_lookup`` in the ``DAR Staging`` database.
    Also stores final model object and feature list in GitHub via pickle. Model objects are compressed into zip files.
    Input Attributes:
        * **model_obj**: An instance of the ``model_setup_fit.Model`` class OR the ``model_soptimization.ModelOptimizer`` class, mandatory.
        * **features**: list of final feature names associated with the final model, mandatory.
        * **name**: Your name (as a string), mandatory
        * **optional keyword arguments**: to be supplied ONLY IF you want to manually populate ``dm.ml_model_results_lookup`` after running a model outside of Python.
            * *model*:  Type of model used, e.g. *'logisticregression'*
            * *yyar*: y variable predicted, e.g. *'bandwidth_in_mbps'*
            * *accuracy_train...*: performance stats on the training set (refer to ``dm.ml_model_results_lookup`` for all the possible options)
            * *accuracy_test...*: same as above for the test set
    """
    featurepath = 'objects/feature_cols/'
    modelpath = 'objects/models/'

    resultsdict = build_resultsdict(model_obj, features, **kwargs)

    # write out features
    with open(featurepath + 'features_' + resultsdict['model_id'] + '.pkl', 'w') as f:
        dump(features.tolist(), f)
    # write out model
    model_pickle = modelpath + 'model_' + resultsdict['model_id'] + '.pkl'
    with open(model_pickle, 'wb') as f:
        dump(model_obj.bestmodel, f, HIGHEST_PROTOCOL)

    # insert into database
    columns = resultsdict.keys()
    values = [resultsdict[column] for column in columns]

    hstore_dict = [val for val in values if type(val) is dict][0]
    values = [val for val in values if type(val) != dict]
    columns = [c for c in columns if c != 'classifier_params']

    conn_ml = psycopg2.connect(host=HOST_FORKED, user=USER_FORKED, password=PASSWORD_FORKED, dbname=DB_FORKED)
    cur_ml = conn_ml.cursor(cursor_factory=DictCursor)

    insert_statement = 'insert into dm.ml_model_results_lookup (%s) values %s;'
    cur_ml.execute(insert_statement, (AsIs(','.join(columns)), tuple(values)))
    print("Results inserted")
    insert_statement_hstore = "update dm.ml_model_results_lookup set classifier_params = (%s) where model_id = '" + resultsdict['model_id'] + "';"
    cur_ml.execute(insert_statement_hstore, [Json(hstore_dict)])
    # insert name and date
    cur_ml.execute("update dm.ml_model_results_lookup set name = '" + name + "' where model_id = '" + resultsdict['model_id'] + "';")
    cur_ml.execute("update dm.ml_model_results_lookup set date = '" + str(pd.to_datetime(datetime.now()).date()) + "'::DATE where model_id = '" + resultsdict['model_id'] + "';")
    if "comment" in kwargs:
        cur_ml.execute("update dm.ml_model_results_lookup set comment = '" + kwargs['comment'] + "' where model_id = '" + resultsdict['model_id'] + "';")

    cur_ml.close()
    conn_ml.commit()
    conn_ml.close()
    print("All results inserted")


def attach_y_post_model(model_obj):
    """After a final model is fit (using ``Model`` or ``ModelOptimizer``), takes in an instance of either class and attaches back the y variable and its predicted value according to the model for post-prediction analysis.
    Outputs are the new feature matrices *X_train* and *X_test* with the 2 additional columns for the actual and predicted y variables."""
    best_imputer = model_obj.bestimputer
    best_model = model_obj.bestmodel
    new_X_train = new_X_test = pd.DataFrame()

    for sets in ['train', 'test']:
        X = getattr(model_obj, 'X_' + sets)
        # apply the best imputer strategy
        Xt = best_imputer.fit_transform(X)
        y = getattr(model_obj, 'y_' + sets)
        y_pred = best_model.predict(Xt)

        if model_obj.yvar != 'fiber_binary' and model_obj.classification_type == 'classificiation':
            Xf = pd.DataFrame(X).reset_index(drop=True)
            yf = pd.Series(model_obj.le.inverse_transform(y)).reset_index(drop=True)
            y_predf = pd.Series(model_obj.le.inverse_transform(y_pred)).reset_index(drop=True)
            exec('new_X_' + sets + ' = pd.concat([Xf,yf,y_predf],axis=1)')
        else:
            Xf = pd.DataFrame(X).reset_index(drop=True)
            yf = pd.Series(y).reset_index(drop=True)
            y_predf = pd.Series(y_pred).reset_index(drop=True)
            exec('new_X_' + sets + ' = pd.concat([Xf,yf,y_predf],axis=1)')

        if isinstance(model_obj, Model):
            cols = model_obj.training_data.columns.tolist()
            cols = cols + ['y', 'y_pred']
        else:
            cols = model_obj.features.tolist() + ['y', 'y_pred']

        exec("new_X_" + sets + ".columns = cols")

    return new_X_train, new_X_test
