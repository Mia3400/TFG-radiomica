import pandas
from numpy import mean
from sklearn.model_selection import train_test_split
from coeficients import SSR,SST,r2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score

# Base-line a la variable OS
clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
data = pandas.read_excel(clinical_data_path)
predict = data.loc[:,["TCIA_ID","OS"]]
predict_data = pandas.DataFrame(predict).set_index("TCIA_ID")

# Carregam les dades processades i separades:
y_train = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Ytrain_processed.csv").set_index("TCIA_ID")
y_test = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Ytest_processed.csv").set_index("TCIA_ID")
prediccions_baseline = []
media = mean(y_train.loc[:,"OS"])
for i in range(0,len(y_train)):
    prediccions_baseline.append(media)

# Avaluació del model:
prediccions_test =[]
for i in range(0,len(y_test)):
    prediccions_test.append(media)

r2_valor= r2_score(y_true=y_train, y_pred=prediccions_baseline)

print("El valor del coeficient de determinació a les dades d'entrenament és", r2_valor)
y_train_lista = list(y_train.loc[:,"OS"])

print("A test val", r2_score(y_pred=prediccions_test,y_true=y_test))
