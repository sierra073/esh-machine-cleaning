import pandas as pd
import numpy as np
import math
from collections import Counter
import pickle

class PreprocessRaw(object):
    """A raw dataset (coming from frns and frn_line_items tables) that can be cleaned and prepped for Machine Cleaning modeling by applying the helper functions in this module.
    """

    """declare the columns to be dropped that could never be useful"""
    drop_cols = ['id', 'frn', 'frn_number_from_the_previous_year', 'application_number', 'ben', 'account_number', 'service_provider_number','establishing_fcc_form470', 'user_entered_establishing_fcc_form470','line_item', 
    'award_date', 'expiration_date', 'contract_expiry_date', 'service_start_date','model','contract_number', 'restriction_citation', 'other_manufacture', 
    'download_speed','download_speed_units','upload_speed','upload_speed_units','burstable_speed','burstable_speed_units','purpose', 'source_of_matching_funds']

    """declare the columns that are yes/no"""
    yn_cols = ['connection_supports_school_library_or_nif', 'includes_voluntary_extensions', 'basic_firewall_protection', 'based_on_state_master_contract',
    'based_on_multiple_award_schedule', 'pricing_confidentiality', 'lease_or_non_purchase_agreement', 'connected_directly_to_school_library_or_nif','was_fcc_form470_posted','frn_previous_year_exists']

    """declare the columns that don't need the float conversion"""
    cat_cols = ['pricing_confidentiality', 'based_on_state_master_contract', 'lease_or_non_purchase_agreement', 'based_on_multiple_award_schedule',  
    'was_fcc_form470_posted', 'connection_supports_school_library_or_nif' ,'connected_directly_to_school_library_or_nif','basic_firewall_protection','includes_voluntary_extensions','pricing_confidentiality_type',
    'fiber_type','connection_used_by','fiber_sub_type','purpose','unit','function','postal_cd','type_of_product','service_provider_name','billed_entity_name',
    'funding_request_nickname','narrative','contact_email','frn_previous_year_exists']

    def __init__(self, df, verbose):
        self.df = df
        self.verbose = verbose

    def getdata(self):
        return self.df

    def remove_column_nulls(self):
        """Removes columns that are all null."""
        if self.verbose==True:
            print("Dropped null columns: ")
            print(self.df.columns[self.df.isnull().all()].tolist())
            self.df = self.df.dropna(axis=1, how='all')
        else:
            self.df = self.df.dropna(axis=1, how='all')
        return self

    def duplicate_columns(self):
        """Obtains the names of the columns that have identical values to any other columns."""
        groups = self.df.columns.to_series().groupby(self.df.dtypes).groups
        dups = []
        for t, v in groups.items():
            dcols = self.df[v].to_dict(orient="list")

            vs = dcols.values()
            ks = dcols.keys()
            lvs = len(vs)

            for i in range(lvs):
                for j in range(i+1,lvs):
                    if vs[i] == vs[j]: 
                        dups.append(ks[i])
                        break

        return dups  

    def remove_column_duplicates(self):
        """Removes the columns that have duplicates (as determined by *duplicate_columns()*).
        Also removes columns that have the same name as other columns in case they aren't caught by the above."""
        dups = self.duplicate_columns()
        self.df = self.df.drop(dups, axis=1)
        self.df = self.df.loc[:,~self.df.columns.duplicated()]
        if self.verbose==True:
            print("Dropped duplicate columns: ")
            print(dups)
        return self


    def summary(self):
        """Takes a DataFrame and creates a summary. Does different things if object or numeric features. 
        Reports the data type, some stats, the number of unique values, and null count/percent for each column.
        """
        summary_list = []
        if self.verbose==True:
            print 'SHAPE', self.df.shape
        
        for i in self.df.columns:
            vals = self.df[i]    
            if self.df[i].dtype == 'O':
                try:
                    most_frequent = Counter(vals[vals != None].tolist()).most_common(1)
                    uniq = vals.nunique()
                except TypeError:
                    most_frequent = 'NA'
                    uniq = 'NA'
                summary_list.append([i,
                                     vals.dtype, 
                                     'NA', 
                                     'NA', 
                                     most_frequent,
                                     uniq, 
                                     sum(pd.isnull(vals)),
                                     sum(pd.isnull(vals))/(1.0*len(self.df))])
            else:
                summary_list.append([i,
                                     vals.dtype, 
                                     vals.min(), 
                                     vals.max(), 
                                     vals.mean(),
                                     vals.nunique(), 
                                     sum(pd.isnull(vals)),
                                     sum(pd.isnull(vals))/(1.0*len(self.df))])
        return pd.DataFrame(summary_list, columns=['col','datatype','min','max','mean_or_most_common','num_uniq','null_count','null_pct'])

    def remove_no_var(self):
        """Removes columns with 0 variance."""
        cols = list(self.df)
        nunique = self.df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        self.df = self.df.drop(cols_to_drop, axis=1)
        if self.verbose==True and len(cols_to_drop) >=1:
            print("Dropped 0-variance columns: ")
            print(cols_to_drop)
        return self

    def remove_drops_raw(self):
        """Removes the columns *drop_cols* we specified to drop for the class"""
        self.df = self.df.drop(self.__class__.drop_cols, axis=1)
        if self.verbose==True:
            print("Dropped: ")
            print(self.__class__.drop_cols)
        return self

    def rename_col(self,col,new_name):
        """Rename a column *col* of the dataframe a new name according to *new_name*"""
        self.df = self.df.rename(columns = {col: new_name})
        if self.verbose==True:
            print("Renamed " + col + " to " + new_name)
        return self

    def convert_floats(self,cat_cols):
        """General function: convert columns that should be decimal numbers (if not already) to floats. 
        Since there are more float variables than categorical/string variables, it takes in a list *cat_cols* of those to exclude.
        """
        for col in self.df.columns: 
            if col not in cat_cols:
                try:
                    self.df[col] = self.df[col].astype(float)
                    if self.verbose==True:
                        print(col + " converted to float")
                except (ValueError, TypeError):
                    print(col + " float conversion failed")
                    continue
        return self

    def convert_floats_raw(self):
        """Convert the PreprocessRaw variables to floats that are not in the class categorical variables."""
        self.convert_floats(self.__class__.cat_cols)
        return self

    def convert_yn(self,col):
        """General function: convert a column *col* of the dataframe that contains 'Yes' or 'No' strings to boolean.
        """
        var_corrected=[]
        for _, row in self.df.iterrows():
            if row[col]=='Yes':
                var_corrected.append(True)
            elif row[col]=='No':
                var_corrected.append(False)
            else:
                var_corrected.append(None)
        self.df[col] = var_corrected
        if self.verbose==True:
            print(col + " converted to boolean")
        return self

    def convert_yn_raw(self):
        """Convert the PreprocessRaw class 'Yes'/'No' variables to boolean."""
        for col in self.__class__.yn_cols:
            self.convert_yn(col)
        return self

    def convert_dummies(self,cols):
        """General function: convert multiple categorical columns specified in *cols* to dummy variables if their cardinality is <= 9 or the column is 'postal_cd'.
        """
        dummy_cols = []
        for col in cols:
            #convert column to lowercase
            self.df[col] = self.df[col].map(lambda x: x if type(x)!=str else x.lower())
            #replace spaces and slashes with underscore
            self.df[col] = self.df[col].replace('\s+', '_',regex=True).replace('/', '_',regex=True)
            #check cardinality
            if (self.df.groupby(col)['download_speed_mbps'].max().reset_index().shape[0] <= 9) or (col == 'postal_cd'):
                dummy_cols.append(col)

        dummy_cols_prefixed = [col.split('_', 1)[0] for col in dummy_cols]
        if self.verbose==True:
            print("Dummified columns: ")
            print(dummy_cols)

        self.df = pd.get_dummies(self.df, columns=dummy_cols, prefix=dummy_cols_prefixed)
        return self

    def convert_dummies_raw(self):
        """Convert the PreprocessRaw class categorical variables to dummies"""
        s = set(self.__class__.yn_cols)
        cat_cols_no_bool = [x for x in self.__class__.cat_cols if x not in s]
        self.convert_dummies(cat_cols_no_bool)
        return self

    def remove_mostly_nulls(self):
        """Remove columns of the dataset that are >= 74% null."""
        s1 = self.summary()
        s1.sort_values('col')
        mostly_not_null_cols = s1[s1.null_pct < .74]
        self.df = self.df[mostly_not_null_cols.col.tolist()]
        if self.verbose == True:
            print("Dropped columns >= 74% NULL: ")
            print(s1[s1.null_pct >= .74].col.tolist())
        return self

    def remove_correlated_raw(self, threshold):
        col_corr = set() # Set of all the names of deleted columns
        data_sub = self.df.loc[:, self.df.dtypes == float]
        #create a dict of the float columns and their number of nulls
        nnull_dict = data_sub.isnull().sum(axis=0).to_dict()
        print(nnull_dict)

        corr_matrix = data_sub.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] >= threshold:
                    colname1 = corr_matrix.columns[i]
                    colname2 = corr_matrix.columns[j]
                    #choose the one with more nulls to remove
                    if nnull_dict[colname1] > nnull_dict[colname2]:
                        colname = colname2
                        othercolname = colname1
                    else:
                        colname = colname1
                        othercolname = colname2
                    col_corr.add(colname)
                    if colname in self.df.columns:
                        self.df = self.df.drop(colname, axis=1) # deleting the column from the dataset
                        if self.verbose == True:
                            print("Dropped " + colname + " due to high correlation with " + othercolname)

    def applyall_raw(self):
        """Apply all functions to a PreprocessRaw dataset to preprocess the raw features."""
        self.remove_column_nulls().remove_column_duplicates().remove_no_var().remove_drops_raw().rename_col('purpose_adj','purpose').convert_floats_raw().convert_yn_raw().convert_dummies_raw().remove_mostly_nulls().remove_correlated_raw(.9)

        return self


def final_columns_pickle(df,filepath):
    """Pickle the final columns of your dataset and export. Only meant to be used once all preprocessing/EDA is finalized before modeling.
    Takes in a dataframe and the string *filepath* as the file path of the export (no extension)."""
    with open(filepath + '.pkl','w') as f:
        pickle.dump(df.columns.tolist(),f)