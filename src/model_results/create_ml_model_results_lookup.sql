DROP TABLE IF EXISTS dm.ml_model_results_lookup;

CREATE TABLE  dm.ml_model_results_lookup (
model_id    VARCHAR ,
y   VARCHAR ,
nfeatures   INTEGER ,
imputer  VARCHAR ,
classifier  VARCHAR ,
classifier_params   JSONB ,
accuracy_train  DECIMAL ,
precision_train DECIMAL ,
recall_train    DECIMAL ,
mse_train   DECIMAL ,
accuracy_test   DECIMAL ,
precision_test  DECIMAL ,
recall_test DECIMAL ,
mse_test    DECIMAL ,
name VARCHAR,
date DATE,
comment VARCHAR) ;
