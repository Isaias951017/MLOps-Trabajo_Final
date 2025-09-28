import pandas as pd
import re
import spacy
from sklearn.preprocessing import LabelEncoder

class PreprocesadorTexto:
    def __init__(self, df, stopwords={
            "medico", "paciente", "psicologo", "psicologa",
            "psicologia", "psicoterapeuta", "psicoterapia", "refiere"}):
        self.df = df.copy()
        self.stopwords = stopwords
        self.nlp = spacy.load("es_core_news_lg")
        self.stopwords = self.nlp.Defaults.stop_words.union(self.stopwords)

    def concatenar_columnas(self, df, columna1="subjetivo", columna2="objetivo", nueva_columna="concatenada"):
        self.df = df
        self.df[nueva_columna] = self.df[columna1].astype(str) + " " + self.df[columna2].astype(str)
        return self.df

    def expresiones_regulares(self, columna):
        self.df[columna] = (
            self.df[columna]
            .astype(str)
            .str.lower()
            .apply(lambda x: re.sub(r'\s+', ' ', re.sub(r'[^a-zñü ]', '', x)).strip())
        )

    def tokenizar(self, columna):
        self.df[columna] = (
            self.df[columna]
            .astype(str)
            .fillna("")
            .apply(lambda x: [
                token.text for token in self.nlp(x)
                if token.text.lower() not in self.stopwords and not token.is_punct and not token.is_space
            ])
        )

    def lematizar(self, columna):
        self.df[columna] = self.df[columna].apply(
            lambda x: [token.lemma_ for token in self.nlp(" ".join(x))] if isinstance(x, list) else []
        )

    def label_encodering(self, columna, nueva_columna, tipo="sexo"):
        label_encoder = LabelEncoder()
        self.df[nueva_columna] = label_encoder.fit_transform(self.df[columna])
        mapping_df = pd.DataFrame({
            tipo.capitalize(): label_encoder.classes_,
            'Codigo': label_encoder.transform(label_encoder.classes_)
        })
        return mapping_df

    def procesar(self, columna_texto, columna_sexo=None, columna_grupo=None):
        self.expresiones_regulares(columna_texto)
        self.tokenizar(columna_texto)
        self.lematizar(columna_texto)

        mappings = {}
        if columna_sexo:
            mappings["sexo"] = self.label_encodering(columna_sexo, "sexo_codificado", tipo="sexo")
        if columna_grupo:
            mappings["grupo"] = self.label_encodering(columna_grupo, "grupo_codificado", tipo="grupo")
        return self.df, mappings