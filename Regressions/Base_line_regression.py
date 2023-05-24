import pandas
from numpy import mean
from sklearn.model_selection import train_test_split
from coeficients import SSR,SST,r2
from sklearn.metrics import mean_squared_error,r2_score

clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
data = pandas.read_excel(clinical_data_path)
predict = data.loc[:,["TCIA_ID","OS"]]
predict_data = pandas.DataFrame(predict).set_index("TCIA_ID")

X_train, X_test, y_train, y_test = train_test_split(data, predict_data, train_size = 0.7,shuffle = True, random_state= 1234)

predicciones_baseline = []
media = mean(y_train.loc[:,"OS"])
for i in range(0,len(y_train)):
    predicciones_baseline.append(media)

predicciones_test =[]
for i in range(0,len(y_test)):
    predicciones_test.append(media)

mse = mean_squared_error(y_true=y_train, y_pred=predicciones_baseline)
r2_or= r2_score(y_true=y_train, y_pred=predicciones_baseline)
# Donen 0 per deficnició 
print(mse,r2_or)
y_train_lista = list(y_train.loc[:,"OS"])

print("A test:")
print(r2_score(y_pred=predicciones_test,y_true=y_test))
#Dona negatiu wtf