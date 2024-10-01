import tensorflow as tf
import numpy as np
import os
import sys

# Adicionar o diretório Generador ao sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Generador'))

from Generate_fake_data import generate_fake_data
from src.data_preprocessing import load_and_preprocess_data
from src.model import create_model, train_model
from src.evaluation import evaluate_model, plot_history
from src.config import DATA_PATH

def main():
    # Configurar o seed para reprodutibilidade
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Gerar dados falsos
    generate_fake_data(DATA_PATH)
    print(f"Fake data generated and saved to {DATA_PATH}")
    
    # Configurar o uso de memória da GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # Carregar e pré-processar os dados
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data()

    # Criar o modelo
    model = create_model(X_train.shape[1])

    # Treinar o modelo
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Avaliar o modelo
    evaluate_model(model, X_test, y_test)

    # Visualizar o histórico de treinamento
    plot_history(history)

if __name__ == "__main__":
    main()