import os
import psycopg2

# these commands retrieve the credentials from your .env file or .bash_profile
HOST = os.environ.get("HOST_FORKED")
USER = os.environ.get("USER_FORKED")
PASSWORD = os.environ.get("PASSWORD_FORKED")
DB = os.environ.get("DB_FORKED")

# iniitalize a connection and a cursor
conn = psycopg2.connect(host=HOST, user=USER, password=PASSWORD, dbname=DB)
cur = conn.cursor()

queryfile = open('create_ml_model_results_lookup.sql', 'r')
query = queryfile.read()
queryfile.close()
cur.execute(query)

cur.close()
conn.commit()
conn.close()
