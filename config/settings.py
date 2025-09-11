import os
from dotenv import load_dotenv

# Cargar variables desde el archivo .env
load_dotenv()

DB_SERVER = os.getenv("DB_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_DRIVER = os.getenv("DB_DRIVER")

DB_CONNECTION_STRING = (
    f"mssql+pyodbc://@{DB_SERVER}/{DB_NAME}"
    f"?driver={DB_DRIVER.replace(' ', '+')}&trusted_connection=yes"
)
