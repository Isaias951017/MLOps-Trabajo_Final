from sqlalchemy import create_engine
from config.settings import DB_CONNECTION_STRING

try:
    engine = create_engine(DB_CONNECTION_STRING)
    with engine.connect() as conn:
        result = conn.execute("SELECT GETDATE();")
        print("✅ Conexión exitosa a SQL Server:", list(result))
except Exception as e:
    print("❌ Error al conectar:", e)
