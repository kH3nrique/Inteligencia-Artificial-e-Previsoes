#P1 -> Import data base
import pandas as pd
from IPython.display import display

table = pd.read_csv("clientes.csv")
display(table.info())
print('\n')
    
#P2 -> Prepare the data base

#import the label enconder
from sklearn.preprocessing import LabelEncoder

#criate and apply the Label Enconder in the columns
condificator = LabelEncoder()
table["profissao"] = condificator.fit_transform(table["profissao"])
table["mix_credito"] = condificator.fit_transform(table["mix_credito"])
table["comportamento_pagamento"] = condificator.fit_transform(table["comportamento_pagamento"])

display(table.info())

#2 division -> who preview and how will use to make new previews
y = table["score_credito"]
x = table.drop(columns=["score_credito", "id_cliente"]) 

#train and test
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

#P3 -> Criate a model of AI (Score Good, standard, poor)
#criate AI, models -> Decision Tree e KNN
#importation e criation:
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelTree = RandomForestClassifier()
modelKNN = KNeighborsClassifier()

#AI train
modelTree.fit(xTrain, yTrain)
modelKNN.fit(xTrain, yTrain)

#models test
previewTree = modelTree.predict(xTest)
previewKNN = modelKNN.predict(xTest)

#P4 -> Pic the best model
from sklearn.metrics import accuracy_score

print(accuracy_score(yTest,previewTree)) #best model
print(accuracy_score(yTest,previewKNN))

#P5 -> Use the AI from new previews
newTable = pd.read_csv("novos_clientes.csv")
display(newTable)

newTable["profissao"] = condificator.fit_transform(newTable["profissao"])
newTable["mix_credito"] = condificator.fit_transform(newTable["mix_credito"])
newTable["comportamento_pagamento"] = condificator.fit_transform(newTable["comportamento_pagamento"])

preview = modelTree.predict(newTable)
print(preview)