# Análise de Inadimplência com Machine Learning

Este projeto implementa um modelo de machine learning para previsão de inadimplência utilizando árvores de decisão. O modelo inclui técnicas avançadas de tratamento de dados desbalanceados e validação cruzada para garantir resultados robustos.

## 🎯 Objetivo

O objetivo principal é prever a probabilidade de inadimplência de clientes utilizando características do histórico automotivo. O projeto implementa um classificador binário que pode identificar potenciais casos de inadimplência.

## 🔧 Técnicas Implementadas

- **Modelo Base**: Árvore de Decisão (Decision Tree Classifier)
- **Tratamento de Dados Desbalanceados**:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - NearMiss (Under-sampling)
- **Validação**: 
  - K-Fold Cross Validation
  - Stratified K-Fold
- **Métricas de Avaliação**:
  - Acurácia
  - Precisão
  - Recall
  - F1-Score
  - Curva ROC e AUC
  - Curva Precision-Recall
  - Matriz de Confusão

## 📊 Características do Modelo

- Profundidade máxima da árvore: 10
- Validação cruzada com 5 folds
- Conjunto de teste: 15% dos dados
- Estratificação mantida durante o split dos dados

## 📋 Requisitos

```python
pandas
scikit-learn
matplotlib
imbalanced-learn
```

## 🚀 Como Executar

1. Instale as dependências:
```bash
pip install pandas scikit-learn matplotlib imbalanced-learn
```

2. Certifique-se de que o arquivo 'emp_automovel.csv' está no diretório do projeto

3. Execute o script principal:
```bash
python first.py
```

## 📈 Resultados

O modelo gera diversas visualizações e métricas:
- Matriz de confusão
- Curva ROC
- Curva Precision-Recall
- Relatório de classificação completo
- Intervalos de confiança para as métricas

## 🔍 Características do Dataset

O dataset (`emp_automovel.csv`) contém informações sobre empréstimos automotivos, incluindo a variável alvo 'inadimplente'.

## ⚖️ Balanceamento de Classes

O projeto implementa duas abordagens para lidar com o desbalanceamento de classes:
1. **Oversampling** com SMOTE: Cria amostras sintéticas da classe minoritária
2. **Undersampling** com NearMiss: Reduz as amostras da classe majoritária

## 📝 Notas Adicionais

- O modelo utiliza pipeline para garantir que não haja vazamento de dados durante o processo de validação cruzada
- Implementação de intervalos de confiança para avaliar a robustez do modelo
- Visualizações detalhadas para análise de performance