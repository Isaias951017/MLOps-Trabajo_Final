from src.preprocessing.functions import expresiones_regulares, tokenizar, lematizar, label_encodering_grupo
import pandas as pd
from sqlalchemy import create_engine, text
from config.settings import DB_CONNECTION_STRING
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import string

# Leer el archivo SQL
with open("sql_queries/queries.sql", "r", encoding="utf-8") as file:
    query = file.read()

params = {
    "medico": "PSICOLOGÍA",
    "fechaini": "2023-01-01",
    "fechafin": "2025-05-04"
}

# Ejecutar consulta
engine = create_engine(DB_CONNECTION_STRING)
with engine.connect() as conn:
    historias_clinicas = pd.read_sql(text(query), conn, params=params)

# n para pruebas
n = 3000

# Procesamiento de texto
start_time = time.time()
processed_data = expresiones_regulares(historias_clinicas.head(n), "Concatenada")
datos_tokenizados = tokenizar(processed_data, "Concatenada")
datos_lematizados = lematizar(datos_tokenizados, "Concatenada")

# Codificación del grupo
categotyGrupo, mapping_grupo = label_encodering_grupo(datos_lematizados, "GRUPO", "Label_Grupo")

# Crear columna con texto lematizado como string
categotyGrupo["Concatenado1"] = categotyGrupo["Concatenada"].apply(lambda x: " ".join(x))

categotyGrupo.to_parquet("datos_modelo.parquet", index=False)




