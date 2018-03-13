from preprocess_raw import *

class PreprocessTransformed(PreprocessRaw):
    """A raw dataset (coming from frns and frn_line_items tables), *as well as additional fields from the pristine line items table* that can be cleaned and prepped for Machine Cleaning modeling by applying the helper functions in this module. 
    Inherits the `PreprocessRaw` class.
    """

    def __init__(self, df, **kwargs):
        PreprocessRaw.__init__(self, df, **kwargs)

    def treat_tough_string_vars(self,col):
        """General function: treat the variable *col* to enable conversion to int.
        Ex) for num_lines there are values such as "1.0", "1", "Unknown", "NaN" that do not allow for easy conversion to type int.
        """
        var_corrected = []
        for row in self.df[col]:
            if row != "Unknown":
                try:
                    var_corrected.append(int(re.split(r"\.\s*", row)[0]))
                except:
                    var_corrected.append(None)
            else:
                var_corrected.append(None)

        self.df[col] = var_corrected
        return self

    def convert_ints_transformed(self):
        """Convert just consortium_shared (for now) to int."""
        def make_int(x):
            """General function: make values integer that are floats/when there is a NULL value."""
            if pd.isnull(x):
                return None
            else:
                return int(float(x))

        self.df.loc[:,'consortium_shared'] = self.df.consortium_shared.apply(make_int)
        #self.df.loc[:,'line_item_id'] = self.df.line_item_id.apply(make_int)
        if self.verbose==True:
            print("consortium_shared converted to Int")
        return self

    def remove_drops_transformed(self):
        """Removes the raw cost, charges and service provider columns and keeps the transformed pristine ones."""
        transformed_cost_cols = ['one_time_elig_cost','rec_cost','rec_elig_cost','total_cost']
        for col in self.df.columns.values:
            if (col.find("_cost") != -1 and col not in transformed_cost_cols) or col.find("_charges") != -1 or col=='service_provider_name':
                self.df = self.df.drop(col, axis=1)
                if self.verbose==True:
                    print(col + " removed")
        return self

    def convert_dummies_transformed(self):
        """Convert the additional transformed categorical variables to dummies."""
        transformed_cat_cols = ['connect_category','connect_type','purposetransformed']
        self.convert_dummies(transformed_cat_cols)
        return self

    def remove_correlated_transformed(self):
        col_corr = set() # Set of all the names of deleted columns
        data_sub = self.df.loc[:, self.df.dtypes == float]
        # create a dict of the float columns and their number of nulls
        nnull_dict = data_sub.isnull().sum(axis=0).to_dict()
        print(nnull_dict)

        corr_matrix = data_sub.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) >= self.corr_threshold:
                    colname1 = corr_matrix.columns[i]
                    colname2 = corr_matrix.columns[j]
                    #choose the one with more nulls to remove
                    if nnull_dict[colname1] > nnull_dict[colname2]:
                        if colname2 not in ['bandwidth_in_mbps','num_lines','rec_cost','total_cost']:
                            colname = colname2
                            othercolname = colname1
                        else:
                            colname = colname1
                            othercolname = colname2
                    else:
                        if colname1 not in ['bandwidth_in_mbps','num_lines','rec_cost','total_cost']:
                            colname = colname1
                            othercolname = colname2
                        else:
                            colname = colname2
                            othercolname = colname1
                    col_corr.add(colname)
                    if colname in self.df.columns:
                        self.df = self.df.drop(colname, axis=1) # deleting the column from the dataset
                        if self.verbose == True:
                            print("Dropped " + colname + " due to high correlation with " + othercolname)

    def applyall_transformed(self):
        """Apply all functions to a `PreprocessTransformed` dataset to preprocess the raw + transformed features."""
        self.remove_row_duplicates().remove_column_nulls().remove_column_duplicates().remove_no_var().remove_drops_raw().rename_col('purpose_adj','purpose').convert_floats_raw().convert_yn_raw().convert_dummies_raw().remove_mostly_nulls().convert_ints_transformed().remove_drops_transformed().convert_dummies_transformed().remove_correlated_transformed()
        return self