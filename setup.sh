# Python 2.7 environment
python -m virtualenv mc_venv
source mc_venv/bin/activate
# environment variables
echo "export GITHUB=/path/to/repo/" >> .bash_profile_mc

echo 'export URL_DAR=""' >> .bash_profile_mc
echo 'export USER_DAR=""' >> .bash_profile_mc
echo 'export PASSWORD_DAR=""' >> .bash_profile_mc
echo 'export HOST_DAR=""' >> .bash_profile_mc
echo 'export DB_DAR=""' >> .bash_profile_mc

echo 'export DB_FORKED=""' >> .bash_profile_mc
echo 'export URL_FORKED=""' >> .bash_profile_mc
echo 'export USER_FORKED=""' >> .bash_profile_mc
echo 'export PASSWORD_FORKED=""' >> .bash_profile_mc
echo 'export HOST_FORKED=""' >> .bash_profile_mc

. .bash_profile_mc

# install dependencies
pip install -r requirements.txt

# build the dm.ml_model_results_lookup table to store model results
cd src/model_results && python create_ml_model_results_lookup.py &&

echo "Done!"
