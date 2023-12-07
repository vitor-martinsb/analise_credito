import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import RandomOverSampler

from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

class DataLoader:

    """
    Classe para carregar e preparar dados para modelos de aprendizado de máquina.

    Parâmetros:
    - file_path: str, default='data/input/'
        O caminho onde os arquivos de dados estão localizados.
    - columns: list, default=[]
        Uma lista de nomes de colunas a serem selecionados dos dados.

    Métodos:
    - get_data(): Retorna os dados de treinamento, dados de teste e variável alvo.
    """

    def __init__(self, file_path='data/input/',columns=[]):
        if columns == []:
            self.data_train = pd.read_csv(file_path+'application_train.csv')
            self.target = self.data_train["TARGET"]
            self.data_train = self.data_train.drop(columns="TARGET")
            self.data_test = pd.read_csv(file_path+'application_test.csv')
        else:
            self.data_train = pd.read_csv(file_path+'application_train.csv')
            self.target = self.data_train["TARGET"]
            self.data_train = self.data_train.drop(columns="TARGET")
            self.data_test = pd.read_csv(file_path+'application_test.csv',usecols=columns)
            self.data_train = self.data_train[columns]

    def get_data(self):
        """
        Retorna os dados de treinamento, dados de teste e variável alvo.

        Retorna:
        - data_train: pd.DataFrame
            Os dados de treinamento.
        - data_test: pd.DataFrame
            Os dados de teste.
        - target: pd.Series
            A variável alvo.
        """
        return self.data_train, self.data_test,self.target

class FeatureEngineering:
    """
    Classe para realizar engenharia de recursos nos dados fornecidos.

    Parâmetros:
    - data_train: pd.DataFrame
        Os dados de treinamento.
    - data_test: pd.DataFrame
        Os dados de teste.
    - target: pd.Series
        A variável alvo.
    """
    def __init__(self, data_train, data_test, target):
        self.data_train = data_train
        self.data_test = data_test
        self.target = target
    
    def engineer_features(self):
        """
        Realiza transformações nos dados, como criação de novas features, tratamento de valores ausentes, etc.

        Retorna:
        - data_train: pd.DataFrame
            Os dados de treinamento após as transformações.
        - target: pd.Series
            A variável alvo após o balanceamento de classes.
        """

        # Codificação de variáveis categóricas em dummies usando ColumnTransformer
        categorical_cols = self.data_train.select_dtypes(include=['object']).columns
        self.encoder = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(drop='first', sparse=False), categorical_cols)],
            remainder='passthrough')

        # Ajustar e transformar o conjunto de treinamento
        self.data_train_encoded = pd.DataFrame(self.encoder.fit_transform(self.data_train))
        
        # Balanceamento de classes usando oversampling
        oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
        X_resampled, y_resampled = oversampler.fit_resample(self.data_train_encoded, self.target)

        # Atualizar os dados e o target após o oversampling
        self.data_train = X_resampled
        self.target = y_resampled

        self.data_train = self.data_train.fillna(0)

        return self.data_train, self.target
    
    def transform_new_data(self, new_data):
        """
        Transforma novos dados aplicando as mesmas transformações realizadas nos dados de treinamento.

        Parâmetros:
        - new_data: pd.DataFrame
            Novos dados a serem transformados.

        Retorna:
        - transformed_data: pd.DataFrame
            Novos dados transformados.
        """
        # Codificar variáveis categóricas usando o mesmo OneHotEncoder
        new_data_encoded = pd.DataFrame(self.encoder.transform(new_data))

        # Preencher valores ausentes com 0
        new_data_encoded = new_data_encoded.fillna(0)

        # Criar um DataFrame com as mesmas colunas que foram usadas nos dados de treinamento
        transformed_data = pd.DataFrame(0, index=new_data.index, columns=self.data_train_encoded.columns)

        # Preencher as colunas correspondentes com os valores codificados
        transformed_data[new_data_encoded.columns] = new_data_encoded

        return transformed_data
    
class ModelTrainer:
    """
    Classe para treinar um modelo de aprendizado de máquina.

    Parâmetros:
    - model: Algoritmo de modelo de aprendizado de máquina
        O modelo a ser treinado.
    - features: pd.DataFrame
        As features de entrada para o modelo.
    - target: pd.Series
        A variável alvo.

    Métodos:
    - train_model(): Treina o modelo e retorna rótulos reais e previstos para avaliação.
    """
    def __init__(self, model, features, target):
        self.model = model
        self.features = features
        self.target = target
    
    def train_model(self):
        """
        Treina o modelo e retorna rótulos reais e previstos para avaliação.

        Retorna:
        - y_test: pd.Series
            Rótulos reais.
        - predictions: pd.Series
            Rótulos previstos pelo modelo.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        return y_test, predictions
    
    def predict_probability(self, data):
        """
        Calcula a probabilidade de um cliente pagar o crédito com base nas características.

        Parâmetros:
        - data: pd.DataFrame
            As características do cliente para as quais a probabilidade deve ser calculada.

        Retorna:
        - probabilities: pd.Series
            Probabilidades calculadas pelo modelo.
        """
        probabilities = self.model.predict_proba(data)[:, 1]
        return probabilities

class ModelEvaluator:
    """
    Classe para avaliar a performance de um modelo.

    Parâmetros:
    - true_labels: pd.Series
        Rótulos reais.
    - predicted_labels: pd.Series
        Rótulos previstos pelo modelo.
    """
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
    
    def evaluate_model(self):
        """
        Avalia o modelo e retorna a precisão e a área sob a curva ROC.

        Retorna:
        - accuracy: float
            Precisão do modelo.
        - roc_auc: float
            Área sob a curva ROC.
        """
        accuracy = accuracy_score(self.true_labels, self.predicted_labels)
        roc_auc = roc_auc_score(self.true_labels, self.predicted_labels)
        return accuracy, roc_auc

if __name__ == "__main__":

    columns_features = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY',
                        'NAME_EDUCATION_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE',
                        'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']
    
    new_data = pd.DataFrame({
        'NAME_CONTRACT_TYPE': ['Cash loans'],
        'CODE_GENDER': ['F'],
        'FLAG_OWN_CAR': ['Y'],
        'FLAG_OWN_REALTY': ['Y'],
        'NAME_EDUCATION_TYPE': ['Higher education'],
        'OCCUPATION_TYPE': ['Accountants'],
        'ORGANIZATION_TYPE': ['Business Entity Type 3'],
        'EXT_SOURCE_1': [0.774761],
        'EXT_SOURCE_2': [0.724],
        'EXT_SOURCE_3': [0.49206]
    })

    data_loader = DataLoader("data/input/",columns = columns_features)
    data_train, data_test, target = data_loader.get_data()

    feature_engineering = FeatureEngineering(data_train, data_test, target)
    features, target = feature_engineering.engineer_features()

    #Random Forest
    print('\n --------------- Random Forest --------------- \n')
    random_forest_model = RandomForestClassifier() # Random Forest
    model_trainer = ModelTrainer(random_forest_model, features, target)
    true_labels, predicted_labels = model_trainer.train_model()
    model_evaluator = ModelEvaluator(true_labels, predicted_labels)
    accuracy, roc_auc = model_evaluator.evaluate_model()
    probabilities = model_trainer.predict_probability(feature_engineering.transform_new_data(new_data))
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Probabilities: {probabilities}")
    print('\n -------------------------------------------------- \n')

    #Logistic_regression_model
    print('\n --------------- Regressão Logistica --------------- \n')
    logistic_regression_model = LogisticRegression() # Regressão Logistica
    model_trainer = ModelTrainer(logistic_regression_model, features, target)
    true_labels, predicted_labels = model_trainer.train_model()
    model_evaluator = ModelEvaluator(true_labels, predicted_labels)
    accuracy, roc_auc = model_evaluator.evaluate_model()
    probabilities = model_trainer.predict_probability(feature_engineering.transform_new_data(new_data))
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Probabilities: {probabilities}")
    print('\n -------------------------------------------------- \n')

    # Gradient Boosting Classifier
    print('\n ---------------- Gradient Boosting Classifier ---------------- \n')
    gradient_boosting_model = GradientBoostingClassifier()  # Gradient Boosting Classifier
    model_trainer = ModelTrainer(gradient_boosting_model, features, target)
    true_labels, predicted_labels = model_trainer.train_model()
    model_evaluator = ModelEvaluator(true_labels, predicted_labels)
    accuracy, roc_auc = model_evaluator.evaluate_model()
    probabilities = model_trainer.predict_probability(feature_engineering.transform_new_data(new_data))
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Probabilities: {probabilities}")
    print('\n ------------------------------------------------------------- \n')


