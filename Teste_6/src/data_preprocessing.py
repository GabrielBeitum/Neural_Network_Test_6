import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .config import DATA_PATH

def add_features(df):
    df['bytes_per_packet'] = df['bytes'] / df['packets'].clip(lower=1)
    df['log_duration'] = np.log1p(df['duration'])
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['is_weekend'] = pd.to_datetime(df['timestamp']).dt.dayofweek.isin([5, 6]).astype(int)
    df['packet_rate'] = df['packets'] / df['duration'].clip(lower=0.1)
    df['byte_rate'] = df['bytes'] / df['duration'].clip(lower=0.1)
    return df

def remove_outliers(df, columns, factor=1.5):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def create_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ],
        n_jobs=-1  # Usar todos os núcleos disponíveis
    )
    return preprocessor

def load_and_preprocess_data():
    # Carregar dados em chunks para economizar memória
    chunk_size = 10000  # Ajuste conforme necessário
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size):
        chunk = add_features(chunk)
        chunks.append(chunk)
    df = pd.concat(chunks, axis=0)
    del chunks  # Liberar memória

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    df = remove_outliers(df, numeric_features)
    
    df['label'] = df['attack_type'].apply(lambda x: 0 if x == 'Benign' else 1)
    
    X = df.drop(columns=['attack_type', 'label', 'timestamp'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    preprocessor = create_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor