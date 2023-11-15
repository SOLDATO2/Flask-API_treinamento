import pandas as pd
from sklearn import tree
import joblib
print("---------Carregando df---------")
df = pd.DataFrame(pd.read_csv("stroke_prediction_dataset.csv")).dropna()
print("---------df carregado---------")

#Treinamento, só precisa rodar 1 vez ou se der algum problema
#Exclui o model.joblib e faz o treinamento (n sei se tem que excluir)

df['Gender'].replace(['Male', 'Female'], [1,-1], inplace=True) # 1 = Homem | -1 = Mulher
df['Smoking Status'].replace(['Non-smoker', 'Formerly Smoked', 'Currently Smokes'], [1,2,3], inplace=True) # 1 = ñ fuma | 2 = fumava casualmente | 3 = fuma
df['Physical Activity'].replace(['Low', 'Moderate', 'High'], [1,2,3], inplace=True) # 1 = baixo | 2 = medio | 3 = alto
df['Family History of Stroke'].replace(['Yes', 'No'], [1,0], inplace=True) # 1 = sim | 0 = nao
df['Diagnosis'].replace(['Stroke', 'No Stroke'], [1,0], inplace=True) # 1 = AVC | 0 = ñ AVC

X = df[['Age', 'Gender', 'Hypertension', 'Heart Disease', 'Average Glucose Level', 'Smoking Status', 'Physical Activity', 'Stroke History', 'Family History of Stroke']].values
y = df[['Diagnosis']].values
model = tree.DecisionTreeClassifier()
model.fit(X,y) # treinamento com X e y do csv

joblib.dump(model, 'model.joblib')
