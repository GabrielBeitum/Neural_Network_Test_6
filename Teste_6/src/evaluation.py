import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
from tabulate import tabulate

def evaluate_model(model, X_test, y_test):
    # Fazer previsões em lotes para economizar memória
    batch_size = 1000
    y_pred = []
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        y_pred.extend(model.predict(batch).flatten())
    y_pred = np.array(y_pred)
    y_pred_classes = np.round(y_pred)

    # Avaliar o modelo
    evaluation_results = model.evaluate(X_test, y_test)
    
    # Extrair métricas relevantes
    metrics = dict(zip(model.metrics_names, evaluation_results))

    # Calcular a matriz de confusão
    cm = confusion_matrix(y_test, y_pred_classes)

    # Calcular o número de arquivos seguros e inseguros
    num_secure = np.sum(y_test == 0)
    num_insecure = np.sum(y_test == 1)

    # Calcular o número de previsões corretas e incorretas
    if cm.shape == (2, 2):
        correct_secure = cm[0, 0]
        incorrect_secure = cm[0, 1]
        correct_insecure = cm[1, 1]
        incorrect_insecure = cm[1, 0]
    elif cm.shape == (1, 1):
        correct_secure = cm[0, 0] if num_secure > 0 else 0
        incorrect_secure = 0
        correct_insecure = cm[0, 0] if num_insecure > 0 else 0
        incorrect_insecure = 0
    else:
        raise ValueError(f"Unexpected confusion matrix shape: {cm.shape}")

    # Preparar os resultados para a tabela
    results = [
        ["Métrica", "Valor"]
    ]
    for metric_name, metric_value in metrics.items():
        results.append([metric_name.capitalize(), f"{metric_value:.4f}"])
    
    results.extend([
        ["Total de arquivos testados", len(y_test)],
        ["Arquivos seguros", num_secure],
        ["Arquivos inseguros", num_insecure],
        ["Arquivos seguros classificados corretamente", correct_secure],
        ["Arquivos seguros classificados incorretamente", incorrect_secure],
        ["Arquivos inseguros classificados corretamente", correct_insecure],
        ["Arquivos inseguros classificados incorretamente", incorrect_insecure]
    ])

    # Imprimir os resultados em formato de tabela
    print(tabulate(results, headers="firstrow", tablefmt="grid"))

    # Imprimir o relatório de classificação
    print('\nRelatório de Classificação:')
    print(classification_report(y_test, y_pred_classes, target_names=['Seguro', 'Inseguro']))

    # Plotar a matriz de confusão
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Seguro', 'Inseguro'], 
                yticklabels=['Seguro', 'Inseguro'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.tight_layout()
    plt.figtext(0.5, 0.01, 'Esta matriz mostra a contagem de previsões corretas e incorretas para cada classe.', wrap=True, horizontalalignment='center', fontsize=10)
    plt.show()

    # Plotar curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC (Receiver Operating Characteristic)')
    plt.legend(loc="lower right")
    plt.figtext(0.5, 0.01, 'A curva ROC ilustra o desempenho do classificador em diferentes limiares. Um AUC mais próximo de 1 indica melhor desempenho.', wrap=True, horizontalalignment='center', fontsize=10)
    plt.show()

    # Plotar curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title('Curva Precision-Recall')
    plt.figtext(0.5, 0.01, 'Esta curva mostra o trade-off entre precisão e recall para diferentes limiares. Uma área maior sob a curva indica melhor desempenho.', wrap=True, horizontalalignment='center', fontsize=10)
    plt.show()

def plot_history(history):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history.history['auc'], label='Treino')
    plt.plot(history.history['val_auc'], label='Validação')
    plt.title('AUC do Modelo')
    plt.xlabel('Época')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history.history['precision'], label='Treino')
    plt.plot(history.history['val_precision'], label='Validação')
    plt.title('Precisão do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisão')
    plt.legend()
    
    plt.tight_layout()
    plt.figtext(0.5, 0.01, 'Estes gráficos mostram a evolução das métricas de desempenho do modelo durante o treinamento. ' 
                           'As linhas azuis representam o conjunto de treino e as laranjas o conjunto de validação.', 
                wrap=True, horizontalalignment='center', fontsize=10)
    plt.show()