import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, RocCurveDisplay, roc_auc_score, PrecisionRecallDisplay
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.under_sampling import NearMiss

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
# plt.show()

print(f'AUC do modelo: {roc_auc_score(y_test, y_previsao)}')

# Curva Precision-Recall
PrecisionRecallDisplay.from_predictions(y_test, y_previsao, name='Modelo de árvore de decisão')
# plt.show()

print(f'AP do modelo: {average_precision_score(y_test, y_previsao)}')

print(classification_report(y_test, y_previsao))

# K-Fold
model = DecisionTreeClassifier(max_depth=10)
kfold = KFold(n_splits=5, shuffle=True, random_state=5)
cv_results = cross_validate(model, x, y, cv=kfold)
print(cv_results)

media = cv_results['test_score'].mean()
print(f'Média da acurácia do modelo: {media}')

desvio_padrao = cv_results['test_score'].std()
print(f'Desvio padrão da acurácia do modelo: {desvio_padrao}')

print(f'Intervalo de confiança da acurácia do modelo: {media - 2 * desvio_padrao} a {min(media + 2 * desvio_padrao, 1)}')

def intervalo_confianca(resultados):
    media = resultados['test_score'].mean()
    desvio_padrao = resultados['test_score'].std()
    print(f'Intervalo de confiança da acurácia do modelo: {media - 2 * desvio_padrao} a {min(media + 2 * desvio_padrao, 1)}')

model = DecisionTreeClassifier(max_depth=10)
kfold = KFold(n_splits=5, shuffle=True, random_state=5)
cv_results = cross_validate(model, x, y, cv=kfold, scoring='recall')
print(cv_results)

print(intervalo_confianca(cv_results))

print(df['inadimplente'].value_counts(normalize=True))

# Stratified K-Fold
model = DecisionTreeClassifier(max_depth=10)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
cv_results = cross_validate(model, x, y, cv=skfold, scoring='recall')

print(intervalo_confianca(cv_results))

# SMOTE
oversample = SMOTE(random_state=5)
x_balanced, y_balanced = oversample.fit_resample(x, y)

print(y_balanced.value_counts(normalize=True))

model = DecisionTreeClassifier(max_depth=10)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
cv_results = cross_validate(model, x_balanced, y_balanced, cv=skfold, scoring='recall')

print(intervalo_confianca(cv_results))

# Pipeline
model = DecisionTreeClassifier(max_depth=10)
pipeline = imbPipeline(steps=[('oversample', SMOTE(random_state=5)), ('modelArvore', model)])

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
cv_results = cross_validate(pipeline, x, y, cv=skfold, scoring='recall')

print(intervalo_confianca(cv_results))

# NearMiss
model = DecisionTreeClassifier(max_depth=10)
pipeline = imbPipeline(steps=[('undersample', NearMiss(version=3)), ('modelArvore', model)])

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
cv_results = cross_validate(pipeline, x, y, cv=skfold, scoring='recall')

print(intervalo_confianca(cv_results))

undersample = NearMiss(version=3)
x_balanced, y_balanced = undersample.fit_resample(x, y)

model = DecisionTreeClassifier(max_depth=10)
model.fit(x_balanced, y_balanced)
y_previsao = model.predict(x_test)

print(classification_report(y_test, y_previsao))
visualizacao = ConfusionMatrixDisplay(confusion_matrix(y_test, y_previsao), 
                                      display_labels=['Não inadimplente', 'Inadimplente'])
visualizacao.plot()
plt.show()

print(f'Acurácia do modelo: {accuracy_score(y_test, y_previsao)}')
print(f'Precisão do modelo: {precision_score(y_test, y_previsao)}')
print(f'Recall do modelo: {recall_score(y_test, y_previsao)}')
print(f'F1 do modelo: {f1_score(y_test, y_previsao)}')

