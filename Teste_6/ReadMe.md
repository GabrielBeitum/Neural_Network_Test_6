# Projeto de Detecção de Ataques de Rede usando Aprendizado de Máquina

## Visão Geral

Este projeto implementa um sistema avançado de detecção de ataques de rede utilizando técnicas de aprendizado de máquina. O objetivo principal é classificar o tráfego de rede como seguro (benigno) ou inseguro (ataque), baseando-se em diversas características do tráfego. Este sistema pode ser uma ferramenta valiosa para profissionais de segurança da informação e administradores de rede.

## Estrutura do Projeto

O projeto está organizado de forma modular para facilitar a manutenção e expansão. Aqui está a estrutura detalhada:

Teste_6/
│
├── main.py                    # Script principal que coordena todo o processo
├── requirements.txt           # Lista de dependências do projeto
├── data/
│   └── fake_network_data.csv  # Dados de rede gerados artificialmente
├── src/
│   ├── __init__.py            # Torna o diretório src um pacote Python
│   ├── config.py              # Configurações globais do projeto
│   ├── data_preprocessing.py  # Funções para pré-processamento de dados
│   ├── model.py               # Definição e treinamento do modelo
│   └── evaluation.py          # Funções para avaliação do modelo
└── Generador/
    └── Generate_fake_data.py  # Script para gerar dados de rede sintéticos


## Componentes Principais

### 1. Geração de Dados Falsos (Generate_fake_data.py)

Este componente é crucial para o desenvolvimento e teste do sistema sem a necessidade de dados reais de rede, que podem ser sensíveis ou difíceis de obter.

- **Funcionalidade**: Gera dados de tráfego de rede sintéticos que imitam padrões reais.
- **Características geradas**: IP de origem/destino, protocolo, portas, número de pacotes, bytes transferidos, duração da conexão e tipo de ataque.
- **Distribuição**: Simula uma proporção realista entre tráfego seguro e inseguro (70% seguro, 30% inseguro).
- **Tipos de ataque**: Inclui ataques comuns como DoS (Negação de Serviço), Probe (Varredura), R2L (Acesso Remoto não Autorizado) e U2R (Escalonamento de Privilégios).

### 2. Pré-processamento de Dados (data_preprocessing.py)

O pré-processamento adequado dos dados é fundamental para o desempenho do modelo de aprendizado de máquina.

- **Carregamento de dados**: Utiliza um sistema de carregamento em chunks para lidar eficientemente com grandes volumes de dados.
- **Engenharia de características**: Cria novas características derivadas para enriquecer o conjunto de dados:
  - bytes_per_packet: Média de bytes por pacote
  - log_duration: Logaritmo da duração (para lidar com distribuições assimétricas)
  - hour: Hora do dia extraída do timestamp
  - is_weekend: Indicador se o tráfego ocorreu no fim de semana
  - packet_rate e byte_rate: Taxas de pacotes e bytes por segundo
- **Remoção de outliers**: Utiliza o método do Intervalo Interquartil (IQR) para remover valores extremos que podem distorcer o aprendizado.
- **Codificação e normalização**: Aplica codificação one-hot para variáveis categóricas e normalização para variáveis numéricas.

### 3. Modelo de Aprendizado de Máquina (model.py)

O coração do sistema é um modelo de rede neural profunda implementado com TensorFlow/Keras.

- **Arquitetura**: 
  - Camadas densas com ativação ReLU
  - Regularização L2 para prevenir overfitting
  - Normalização em lote para estabilizar o aprendizado
  - Dropout para melhorar a generalização
- **Compilação**:
  - Otimizador: Adam com taxa de aprendizado adaptativa
  - Função de perda: Binary crossentropy (adequada para classificação binária)
  - Métricas: Acurácia, AUC, Precisão e Recall
- **Treinamento**:
  - Implementa early stopping para evitar overfitting
  - Utiliza redução da taxa de aprendizado quando o desempenho estagna

### 4. Avaliação do Modelo (evaluation.py)

Uma avaliação abrangente é essencial para entender o desempenho e as limitações do modelo.

- **Métricas calculadas**:
  - Acurácia: Proporção geral de previsões corretas
  - AUC (Area Under the Curve): Capacidade do modelo de distinguir entre classes
  - Precisão: Proporção de verdadeiros positivos entre todos os positivos previstos
  - Recall: Proporção de verdadeiros positivos identificados corretamente
- **Visualizações**:
  - Matriz de confusão: Mostra a distribuição de previsões corretas e incorretas
  - Curva ROC: Ilustra o desempenho do classificador em diferentes limiares
  - Curva Precision-Recall: Mostra o trade-off entre precisão e recall
- **Relatório detalhado**: Fornece uma tabela com estatísticas detalhadas sobre o desempenho do modelo

### 5. Script Principal (main.py)

Este script orquestra todo o fluxo de trabalho do projeto:

1. Configura o ambiente (seed aleatório, uso de GPU)
2. Gera dados falsos se necessário
3. Carrega e pré-processa os dados
4. Cria e treina o modelo
5. Avalia o modelo e gera visualizações
6. Salva os resultados e o modelo treinado

## Como Executar o Projeto

1. Clone o repositório:
   ```
   git clone [URL_DO_REPOSITÓRIO]
   cd Teste_6
   ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Gere os dados falsos (se necessário):
   ```
   python Generador/Generate_fake_data.py
   ```

4. Execute o script principal:
   ```
   python main.py
   ```

5. Os resultados, incluindo gráficos e métricas de desempenho, serão exibidos e salvos na pasta `results/`.

## Análise e Insights

- **Abordagem de aprendizado profundo**: O uso de redes neurais permite capturar padrões complexos nos dados de rede.
- **Dados sintéticos**: A geração de dados falsos permite o desenvolvimento e teste inicial do sistema, mas é importante validar com dados reais quando possível.
- **Pré-processamento robusto**: A engenharia de características e o tratamento de outliers são cruciais para melhorar o desempenho do modelo.
- **Prevenção de overfitting**: Técnicas como regularização L2, dropout e early stopping ajudam o modelo a generalizar melhor.
- **Avaliação abrangente**: O uso de múltiplas métricas e visualizações fornece uma visão completa do desempenho do modelo.

## Possíveis Melhorias e Expansões Futuras

1. **Validação cruzada**: Implementar k-fold cross-validation para uma avaliação mais robusta do modelo.
2. **Experimentação com arquiteturas**: Testar modelos como RNNs (para capturar dependências temporais) ou CNNs (para padrões espaciais nos dados).
3. **Feature engineering avançada**: Incorporar conhecimento de domínio de segurança de rede para criar características mais sofisticadas.
4. **Balanceamento de classes**: Implementar técnicas como SMOTE se os dados forem muito desbalanceados.
5. **Detecção de anomalias**: Adicionar um componente de detecção de anomalias para identificar novos tipos de ataques não vistos durante o treinamento.
6. **Interpretabilidade do modelo**: Implementar técnicas como SHAP values para entender melhor as decisões do modelo.
7. **Atualização contínua**: Desenvolver um sistema para retreinar o modelo periodicamente com novos dados.
8. **Integração com sistemas de rede**: Criar interfaces para integrar o modelo com sistemas de monitoramento de rede em tempo real.

## Configuração

As configurações principais do projeto estão no arquivo `src/config.py`. Você pode ajustar o caminho do arquivo de dados e outras configurações globais neste arquivo.

## Conclusão

Este projeto demonstra uma abordagem abrangente e moderna para a detecção de ataques de rede usando aprendizado de máquina. Ele oferece uma base sólida que pode ser expandida e refinada para aplicações práticas em ambientes de rede reais. A combinação de geração de dados sintéticos, pré-processamento robusto, modelagem avançada e avaliação detalhada cria um sistema versátil e potente para enfrentar os desafios de segurança cibernética atuais.

## Contribuições

Contribuições para este projeto são bem-vindas! Por favor, leia o arquivo CONTRIBUTING.md para detalhes sobre nosso código de conduta e o processo para enviar pull requests.

## Licença

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo [LICENSE.md](LICENSE.md) para obter detalhes.

## Contato

Para questões ou sugestões, por favor abra uma issue no repositório do GitHub ou entre em contato com Gabriel Marcello Beitum em GabrielBeitum944@hotmail.com.

## Requisitos do Sistema

- Python 3.7 ou superior
- TensorFlow 2.x
- Pandas
- Scikit-learn
- Matplotlib
- NumPy
- Seaborn
- Tabulate

Todas as dependências estão listadas no arquivo `requirements.txt`.
