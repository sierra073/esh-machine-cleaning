from preprocess_raw import *

class PreprocessTransformed(PreprocessRaw):
    def __init__(self, df):
        PreprocessRaw.__init__(self, df)

    def make_int(x):
        """General function: make values integer when there is a NULL value"""
        if pd.isnull(x):
            return None
        else:
            return int(float(x))

    def treat_tough_string_vars(self,col):
        """General function: treat the variable *col* to enable conversion to int
        Ex) num_lines, service_category and bandwidth_in_mbps 
        there are values such as "1.0", "1", "Unknown", "NaN" that do not allow for easy conversion to type int.
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
        """Convert num_lines and service_category to integers."""
        self.df.loc[:,'service_category'] = self.df.service_category.apply(make_int)
        self.treat_tough_string_vars("num_lines")
        return self