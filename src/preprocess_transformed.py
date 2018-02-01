from preprocess_raw import *

class PreprocessTransformed(PreprocessRaw):
    """A raw dataset (coming from frns and frn_line_items tables), *as well as additional fields from the pristine line items table* that can be cleaned and prepped for Machine Cleaning modeling by applying the helper functions in this module. 
    Inherits the PreprocessRaw class.
    """

    def __init__(self, df):
        PreprocessRaw.__init__(self, df)

    """declare the columns to be dropped that could never be useful"""
    drop_cols_transformed = []


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
        return self

    def remove_drops_transformed(self):
        """Removes the raw cost, charges and service provider columns and keeps the transformed pristine ones."""
        transformed_cost_cols = ['one_time_elig_cost','rec_cost','rec_elig_cost','total_cost']
        for col in self.df.columns.values:
            if (col.find("_cost") != -1 and col not in transformed_cost_cols) or col.find("_charges") != -1 or col=='service_provider_name':
                print col
                self.df = self.df.drop(col, axis=1)
        return self

    def convert_dummies_transformed(self):
        """Convert the additional transformed categorical variables to dummies."""
        transformed_cat_cols = ['connect_category','connect_type','purposetransformed']
        self.convert_dummies(transformed_cat_cols)
        return self

    def applyall_transformed(self):
        """Apply all functions to a PreprocessTransformed dataset to preprocess the raw + transformed features."""
        self.applyall_raw().convert_ints_transformed().remove_drops_transformed().convert_dummies_transformed()
        return self