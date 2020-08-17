from datetime import date
from preprocess_raw import *
from preprocess_transformed import *
from model_setup_fit import *
from model_optimization import *
import warnings
warnings.filterwarnings("ignore")
pd.get_option("display.max_rows",999)
pd.get_option("display.max_columns",999)
pd.get_option("display.width",None)

GITHUB = os.environ.get("GITHUB")
sys.path.insert(0, GITHUB + 'Machine_Cleaning_2019/modeling/src')

import time

def print_time(start, end):
    t_sec = round(end - start)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))

## loading pickle/assumes you've already generated features and saved them to pickle  
training_data_transformed = pd.read_pickle(GITHUB+"/Machine_Cleaning_2019/modeling/src/training_data_transformed.pkl")

start = time.time()
## Initialize Model object and pipeline
model = Model(training_data_transformed, 'purpose', 'gradientboosting', 'classification', imputer_strategy=['mean'])
model = model.build_pipe(n_estimators = [500],
                         min_samples_leaf=[2],
                         min_samples_split =[2],
                         max_depth = [36,44],
                         max_features = ['auto'],
                         learning_rate = [0.1],
                         subsample = [0.75])

end = time.time()
print("\n")
print_time(start, end)

print("*** STARTING GRIDSEARCH ***")
## Find the best model parameters
model.fit()

mo_final = ModelOptimizer(model, 'importance', threshold = 0.00001993)
mo_final.optimize()

output_results(mo_final, mo_final.getfeatures(),'Jamie',comment='feature elim gradient boost')
print("*** FINISHED ***")
