import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn import  model_selection
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from coeficients import calculate_aic
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import seaborn as sns

def print_table(data):
    print("{:<50s}{:<10s}".format("Name", "Coefficient"))
    for coef, name in data:
        print("{:<50s}{:<10f}".format(name, coef))
#Regressió lineal només amb les variables clíniques del pacient
#Dades:
#===============================================================================
def predict_feature(path_clinical_data, names):
    predict = []
    clinical_data = pandas.read_excel(path_clinical_data)
    for i in range (0,len(clinical_data)):
        if clinical_data.loc[i,"TCIA_ID"] in names:
            predict.append(clinical_data.loc[i,["TCIA_ID","OS"]])
    predict_data = pandas.DataFrame(predict).set_index("TCIA_ID")
    return predict_data
#Extracció de dades:

clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
X_train = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Xtrain_processed.csv").set_index("TCIA_ID")
X_test = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Xtest_processed.csv").set_index("TCIA_ID")
y_train = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Ytrain_processed.csv").set_index("TCIA_ID")
y_test = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Ytest_processed.csv").set_index("TCIA_ID")

#detectam variables qualificatives, serán just les mateixes que al clinical data
for col in X_train.columns:
    unique_vals = X_train[col].unique()
    if len(unique_vals) <= 7:
        X_train.loc[:,col] = X_train.loc[:,col].astype("category")
print(X_train.head)
X_train.info()   

#Valors NA's
print(X_train.columns[X_train.isna().any()])
X_train = X_train.drop(X_train.columns[X_train.isna().any()],1)

#Estandardització:
for colname in X_test.columns:
    if colname not in X_train.columns:
        X_test = X_test.drop(colname, axis = 1)

X_train_num = X_train.select_dtypes(include=['float64', 'int'])
col_num = X_train_num.columns


scaler = StandardScaler()
X_train.loc[:,col_num] = scaler.fit_transform(X_train.loc[:,col_num])
X_test.loc[:,col_num] = scaler.fit_transform(X_test.loc[:,col_num])

regressio= LinearRegression()
regressio.fit(X_train,y_train)
#Avaluació del model
#=================================================================================
# metriques al conjunt d'entrenament

scoring = "r2"
results = cross_val_score(regressio, X_train, y_train, cv=4, scoring=scoring)
print(results)
print("El coeficient de determinació a les dadesd d'entrenament val: ", results.mean())

#Anàlisi de resiuds:
cv_predicciones = cross_val_predict(
                    estimator = regressio,
                    X         = X_train,
                    y         = y_train,
                    cv        = 4
                 )
cv_predicciones_list= []
for lista in cv_predicciones:
    cv_predicciones_list.append(lista[0])
cv_predicciones_list = np.asarray(cv_predicciones_list)

# Gràfics
# ==============================================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))

axes[0, 0].scatter(y_train, cv_predicciones, edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 0].plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    'k--', color = 'black', lw=2
)
axes[0, 0].set_title('Valor predit vs valor real', fontsize = 10, fontweight = "bold")
axes[0, 0].set_xlabel('Real')
axes[0, 0].set_ylabel('Predicció')
axes[0, 0].tick_params(labelsize = 7)

axes[0, 1].scatter(list(range(len(y_train))), y_train.loc[:,"OS"].tolist() - cv_predicciones_list,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[0, 1].set_title('Residus del model', fontsize = 10, fontweight = "bold")
axes[0, 1].set_xlabel('id')
axes[0, 1].set_ylabel('Residu')
axes[0, 1].tick_params(labelsize = 7)

sns.histplot(
    data    = y_train.loc[:,"OS"].tolist() - cv_predicciones_list,
    stat    = "density",
    kde     = True,
    line_kws= {'linewidth': 1},
    color   = "firebrick",
    alpha   = 0.3,
    ax      = axes[1, 0]
)

axes[1, 0].set_title('Distribució dels residus del model', fontsize = 10,
                     fontweight = "bold")
axes[1, 0].set_xlabel("Residu")
axes[1, 0].set_ylabel("densitat")
axes[1, 0].tick_params(labelsize = 7)


sm.qqplot(
    y_train.loc[:,"OS"].tolist() - cv_predicciones_list,
    fit   = True,
    line  = 'q',
    ax    = axes[1, 1], 
    color = 'firebrick',
    alpha = 0.4,
    lw    = 2
)
axes[1, 1].set_title('Q-Q residus del model', fontsize = 10, fontweight = "bold")
axes[1, 1].set_xlabel("Quantils teòrics")
axes[1, 1].set_ylabel("Quantils mostrals")
axes[1, 1].tick_params(labelsize = 7)

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Diagnòstic de residus', fontsize = 12, fontweight = "bold")
plt.show()

#Dades test
prediccions = regressio.predict(X_test)
prediccions_list= []
for lista in prediccions:
    prediccions_list.append(lista[0])
prediccions_list = np.asarray(prediccions_list)
df_predicciones = pandas.DataFrame({'OS' : y_test.loc[:,"OS"], 'prediccio' : prediccions_list})
print(df_predicciones)

r2 = r2_score(y_true=y_test, y_pred=prediccions)
aic = calculate_aic(regressio, X_test, y_test)
print("El valor de l'AIC és", aic)
print("El coeficient de determinació a les dades test és", r2)

#Gràfic a les dades test:
plt.scatter(y_test, prediccions, edgecolors=(0, 0, 0), alpha = 0.4)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'k--',  lw=2
)
plt.title('Valor predit vs valor real', fontsize = 10, fontweight = "bold")
plt.xlabel('Real')
plt.ylabel('Predicció')
plt.tick_params(labelsize = 7)
plt.show()