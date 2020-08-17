# Machine Cleaning at EducationSuperHighway (ESH)

The Machine Cleaning project develops classification models to predict _corrected USAC Form 471 line item field values_ for the most commonly edited line item fields by ESH Business Analysts. Past analyses were done to determine which fields were the the best "low-hanging fruit" for machine learning; i.e., where machine learning could have the biggest impact in terms of time saved for manual cleaning. These fields are **`Purpose`** and **`Connect Category`**.

## Workflow/Code Architecture Illustration:
These modules were built to facilitate the model building [workflow](https://educationsuperhighway.atlassian.net/wiki/spaces/SA/pages/393609532/Modeling+Process) for this specific ESH application, and to make it easier to use for analysts with a limited machine learning background. The preprocessing module is specific to ESH, and the modeling modules are built as wrappers around `sklearn` modules. These can and should be extended, with the first priority being to rewrite them to work in Python 3+!

## Files and Descriptions:

#### 1. _Environment setup_
| Directory       | File          | Description  |
| ------------- |:-------------:| :-----|
| .      | [setup.sh](setup.sh)     |   Before running this script, go in and edit it for your system. The GITHUB and _FORKED variables need to be edited for your individual system. Then run `./setup.sh && source mc_venv/bin/activate && . ~/.bash_profile_mc` in your command line to set up and activate the environment.  |

#### 2. _Loading the datasets_

| Directory       | File          | Description  |
| ------------- |:-------------:| :-----|
| src/sql       | [get_data_2019_train.sql](src/modeling/sql/get_data_2019_train.sql)     |   This data is used to train the model. Gets the raw USAC FRN & line item data for 2019.  |
| src/sql       | [get_yvar_dar_prod.sql](src/modeling/sql/get_yvar_dar_prod.sql)     |   Labels to train supervised machine learning model to predict purpose or connect category. |
| src/sql       | [get_data_future_predict.sql](src/modeling/sql/get_data_future_predict.sql)     |   Blank file, but this will look similar to `get_data_2019_train.sql` but will pull in the future year's data to predict on. |

#### 3. _Preprocessing the dataset_

| Directory       | File          | Description  |
| ------------- |:-------------:| :-----|
| src      | [preprocess_raw.py](src/modeling/preprocess_raw.py) |  Includes all data preprocessing functions such as removing nulls, duplicates and data conversions to numeric or dummy variables. Also includes function to remove correlated columns. More detail on [ReadtheDocs](https://esh-machine-cleaning-preprocessing.readthedocs.io/en/latest/source/preprocess_raw.html#module-preprocess_raw)|

#### 4. _Training models_

| Directory       | File          | Description  |
| ------------- |:-------------:| :-----|
| src      | [train_demo.ipynb](src/examples/training_demo.ipynb) | Notebook for training and iterating on models. There is a basic demo of the end-to-end process with a Random Forest model. |
| src      | [model_setup_fit.py](src/modeling/model_setup_fit.py) | On [ReadtheDocs](https://esh-machine-cleaning-preprocessing.readthedocs.io/en/latest/source/model_setup_fit.html#module-model_setup_fit) |
| src      | [model_optimization.py](src/modeling/model_optimization.py) | On [ReadtheDocs](https://esh-machine-cleaning-preprocessing.readthedocs.io/en/latest/source/model_optimization.html#module-model_optimization) |

#### 5. _Making Predictions on Purpose and Connect Category using the trained models_

| Directory       | File          | Description  |
| ------------- |:-------------:| :-----|
| src      | [final_update.ipynb](src/examples/apply_models.ipynb) |  Run this notebook to call the `load_and_predict()` function. This function loads in a model and features and applies it to new data to make predictions on purpose and connect category. <br> <br>**Output:** <br> `/data/ml_mass_update.csv` <br> <br>_Note:_ Must input the data frame to predict on (after the minimal preprocessing) and a model id (string) |
