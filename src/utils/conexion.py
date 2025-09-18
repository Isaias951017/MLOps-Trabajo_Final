import pyodbc
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd

class SQLConnection:
    def __init__(self,sql_path, db_server=os.getenv("DB_SERVER"), db_name=os.getenv("DB_NAME"), db_driver=os.getenv("DB_DRIVER"),params={"medico": "PSICOLOGÍA","fechaini": "20230101","fechafin": "20250504"}):
        load_dotenv()
        self.db_name = db_name
        self.db_server = db_server
        self.db_driver = db_driver
        self.params = params
        self.sql_path = sql_path
        
    def get_connection_string(self):
        db_connection_string = (f"mssql+pyodbc://@{self.db_server}/{self.db_name}"f"?driver={self.db_driver.replace(' ', '+')}&trusted_connection=yes")
        return db_connection_string
    
    def create_engine_connection(self):
        connection_string = self.get_connection_string()
        engine = create_engine(connection_string)
        return engine
    
    def generate_dataframe(self):
        with open(self.sql_path, "r", encoding="utf-8") as file: 
            query = file.read()
        
        engine = self.create_engine_connection()
        with engine.connect() as connection:
            df = pd.read_sql(text(query), connection, params=self.params)
        return df