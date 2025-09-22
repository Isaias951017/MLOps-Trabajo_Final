from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

class Train:
    def __init__(self, df, target_column, model, test_size=0.3, training_columns=["concatenada", "sexo_codificado"]):
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.model = model
        self.training_columns = training_columns

    def train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.df[self.training_columns], self.df[self.target_column], test_size=self.test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def vectorizer(self, X_train, X_test, vectorizer_model=TfidfVectorizer(), columns_to_vectorize="concatenada"):
        self.vectorizer_model = vectorizer_model
        self.X_train = X_train
        self.X_test = X_test
        X_train_vect = self.vectorizer_model.fit_transform(X_train[columns_to_vectorize])
        X_test_vect = self.vectorizer_model.transform(X_test[columns_to_vectorize])
        return X_train_vect, X_test_vect

    def create_pipeline_train(self):
        pipeline = Pipeline(steps=[("classifier", self.model)])
        return pipeline

    def train(self):
        X_train, X_test, y_train, y_test = self.train_test_split()
        X_train_vect, X_test_vect = self.vectorizer(X_train=X_train, X_test=X_test)
        pipeline = self.create_pipeline_train()
        pipeline.fit(X_train_vect, y_train)
        return X_test_vect, pipeline