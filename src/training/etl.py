
class EDAdataset():
    
    def __init__(self, df):
        # Crea una copia del dataframe original para evitar modificarlo directamente
        self.df = df.copy()
    
    def lowercase_columns(self):
        # Convierte los nombres de las columnas a minúsculas
        columnames = self.df.columns.str.lower()
        self.df.columns = columnames  # Convert columns to lowercase
    
    def remove_nulls(self):
        # Eliminar filas donde ambas columnas "subjetivo" y "objetivo" son nulas
        self.df = self.df.dropna(subset=["subjetivo", "objetivo"], how="all") # Eliminar filas donde ambas columnas son nulas 
        return self.df

    def capitalize_grupos(self, column_name="grupo"):
        # Capitaliza la primera letra de cada palabra en la columna "grupo"
        self.df[column_name] = self.df[column_name].str.capitalize()
        return self.df
    
    def dataset_eda(self, df):
        # Realiza todas las transformaciones de EDA en el dataframe
        self.df = df
        self.lowercase_columns()
        self.df = self.remove_nulls()
        self.df = self.capitalize_grupos()
        return self.df