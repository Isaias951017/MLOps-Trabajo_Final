import re 
import pandas as pd
import spacy
from spacy.lang.es.stop_words import STOP_WORDS as stopwords
from sklearn import preprocessing



def expresiones_regulares(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    df = df.copy()
    df[columna] = (
        df[columna]
        .astype(str)
        .str.lower()
        .apply(lambda x: re.sub(r'\s+', ' ', re.sub(r'[^a-zñü ]', '', x)).strip())
    )
    return df



def tokenizar(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    df = df.copy()
    nlp = spacy.load("es_core_news_lg")

    stopwords = nlp.Defaults.stop_words
    stopwords_personalizados = [
        "medico", "paciente", "psicologo", "psicologa",
        "psicologia", "psicoterapeuta", "psicoterapia", "refiere"
    ]
    stopwords.update(stopwords_personalizados)

    df[columna] = (
        df[columna]
        .astype(str)
        .fillna("")
        .apply(lambda x: [
            token.text for token in nlp(x)
            if token.text.lower() not in stopwords and not token.is_punct and not token.is_space
        ])
    )
    return df



def lematizar(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    df = df.copy()
    nlp = spacy.load("es_core_news_lg")

    df[columna] = df[columna].apply(
        lambda x: [token.lemma_ for token in nlp(" ".join(x))] if isinstance(x, list) else []
    )
    return df



def label_encodering_sexo(df, columna: str, nueva_columna: str):
    # crear el codificador
    label_encoder = preprocessing.LabelEncoder()
    
    # Ajustar y transformar la columna
    df[nueva_columna] = label_encoder.fit_transform(df[columna])
    
    # crear el dataframe con el mapeo
    mapping_sexo = pd.DataFrame({
        'Sexo': label_encoder.classes_,
        'Codigo': label_encoder.transform(label_encoder.classes_)
    })
    return df, mapping_sexo

def label_encodering_grupo(df, columna: str, nueva_columna: str):
    # crear el codificador
    label_encoder = preprocessing.LabelEncoder()
    
    # Ajustar y transformar la columna
    df[nueva_columna] = label_encoder.fit_transform(df[columna])
    
    # crear el dataframe con el mapeo
    mapping_grupo = pd.DataFrame({
        'Grupo': label_encoder.classes_,
        'Codigo': label_encoder.transform(label_encoder.classes_)
    })
    return df, mapping_grupo
