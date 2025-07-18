import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, RocCurveDisplay, roc_auc_score, PrecisionRecallDisplay
from sklearn.metrics import average_precision_score, classification_report

df = pd.read_csv('emp_automovel.csv')

print(df.head())

# Dividindo as variáveis explicativas e a variável resposta
x = df.drop('inadimplente', axis=1)
y = df['inadimplente'] # Variável resposta

# Criando o modelo de árvore de decisão
model = DecisionTreeClassifier()
model.fit(x, y)
print(f'Acurácia do modelo: {model.score(x, y)}')

# Separando os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, stratify=y, random_state=5)


# Treinando o modelo
model = DecisionTreeClassifier(max_depth=10)
model.fit(x_train, y_train)

# Avaliando o modelo no teste e no treino
print(f'Acurácia de teste do modelo: {model.score(x_test, y_test)}')
print(f'Acurácia de treino do modelo: {model.score(x_train, y_train)}')

# Matriz de confusão
y_previsao = model.predict(x_test)
print(confusion_matrix(y_test, y_previsao))

# Visualização da matriz de confusão
visualizacao = ConfusionMatrixDisplay(confusion_matrix(y_test, y_previsao), 
                                      display_labels=['Não inadimplente', 'Inadimplente'])
visualizacao.plot()
# plt.show()

print(f'Acurácia do modelo: {accuracy_score(y_test, y_previsao)}')
print(f'Precisão do modelo: {precision_score(y_test, y_previsao)}')
print(f'Recall do modelo: {recall_score(y_test, y_previsao)}')
print(f'F1 do modelo: {f1_score(y_test, y_previsao)}')

# Curva ROC
RocCurveDisplay.from_predictions(y_test, y_previsao, name='Modelo de árvore de decisão')
plt.show()

print(f'AUC do modelo: {roc_auc_score(y_test, y_previsao)}')

# Curva Precision-Recall
PrecisionRecallDisplay.from_predictions(y_test, y_previsao, name='Modelo de árvore de decisão')
plt.show()

print(f'AP do modelo: {average_precision_score(y_test, y_previsao)}')

print(classification_report(y_test, y_previsao))

