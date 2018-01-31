preprocess\_raw module
======================

.. automodule:: preprocess_raw
    :members:
    :undoc-members:
    :show-inheritance:

Examples
---------
Below are some examples for how to use this module in isolation, but please see the example starter script on GitHub. Assumes you have a pandas dataframe `data` to start with. `var_yn` is an arbitrary Yes/No variable, and `var_cat1` and `var_cat2` are arbitrary categorical variables in the below examples. ::

    cleanraw = PreprocessRaw(data)
    print("Summary of data")
    cleanraw.summary()

    cleanraw.convert_floats(['var_cat1','var_cat2'])
    cleanraw.convert_yn('var_yn')
    cleanraw.convert_dummies(['var_cat1','var_cat2'])

    cleanraw.applyall_raw()

    final_data = cleanraw.getdata()
