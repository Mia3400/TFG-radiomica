import pandas
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn import  model_selection
from sklearn.model_selection import train_test_split
import numpy as np
from coeficients import r2, calculate_aic
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import seaborn as sns

def predict_feature(path_clinical_data, names):
    predict = []
    clinical_data = pandas.read_excel(path_clinical_data)
    for i in range (0,len(clinical_data)):
        if clinical_data.loc[i,"TCIA_ID"] in names:
            predict.append(clinical_data.loc[i,["TCIA_ID","OS"]])
    predict_data = pandas.DataFrame(predict).set_index("TCIA_ID")
    return predict_data
#A aquest document es realitzarán regressions amb característiques clíniques que destaquen mèdicamtn per intentar millorar el model fet
clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
clinical_data_X_train = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Xtrain_processed.csv").set_index("TCIA_ID")
clinical_data_X_test = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Xtest_processed.csv").set_index("TCIA_ID")
clinical_data_y_train = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Ytrain_processed.csv").set_index("TCIA_ID")
clinical_data_y_test = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Ytest_processed.csv").set_index("TCIA_ID")
clinical_data = pandas.read_excel(clinical_data_path)
predict_data = clinical_data.loc[:,["OS", "TCIA_ID"]].set_index("TCIA_ID")

# Funció que reprodueix el mateix procés que a la regressió original només amb un grup de variables donades
def seleccio_columnes(columnes, clinical_data_X_train,clinical_data_X_test): 
    taula = []
    dades_columna = pandas.DataFrame()
    for nom in columnes:
        dades = []
        for var in clinical_data_X_train.columns:
            if nom in var:
                dades.append(var)
        dades_columna_X_train = clinical_data_X_train.loc[:,dades]
        dades_columna_X_test = clinical_data_X_test.loc[:,dades]
        names_Xtrain= dades_columna_X_train.index.to_list()
        names_Xtest= dades_columna_X_test.index.to_list()                           #Llista amb l'ID del pacients que tenen informacio necessaria
        y_train = predict_feature(clinical_data_path, names_Xtrain)     #OS dels mateixos pacients
        y_test = predict_feature(clinical_data_path, names_Xtest)
        r2_score , Aic , nom_columna = fer_regressio_una_columna(nom,dades_columna_X_train,dades_columna_X_test,y_train, y_test)
        taula.append({
            'Columna': nom_columna,
            'r2_score': r2_score,
            'Aic': Aic
        })
    df = pandas.DataFrame(taula)
    return df.set_index("Columna")


def fer_regressio_una_columna(nom_columna,dades_columna_X_train, dades_columna_X_test,predict_data_y_train,predict_data_y_test):
    for col in dades_columna_X_train.columns:
        unique_vals = dades_columna_X_train[col].unique()
        if len(unique_vals) <= 7:
            dades_columna_X_train.loc[:,col] = dades_columna_X_train.loc[:,col].astype("category")
    X_train_num = dades_columna_X_train.select_dtypes(include=['float64', 'int'])
    col_num = X_train_num.columns
    if len(col_num)!=0:
        scaler = StandardScaler()
        dades_columna_X_train.loc[:,col_num] = scaler.fit_transform(dades_columna_X_train.loc[:,col_num])
        dades_columna_X_test.loc[:,col_num] = scaler.fit_transform(dades_columna_X_test.loc[:,col_num])
        
    regressio= LinearRegression()
    regressio.fit(dades_columna_X_train,predict_data_y_train)
    scoring = "r2"
    results = cross_val_score(regressio, dades_columna_X_train, predict_data_y_train, cv=4, scoring=scoring)
    print(results)
    print("R squared val ", nom_columna, results.mean())

    cv_predicciones = cross_val_predict(
                        estimator = regressio,
                        X         = dades_columna_X_train,
                        y         = predict_data_y_train,
                        cv        = 4
                    )
    cv_predicciones_list= []
    for lista in cv_predicciones:
        cv_predicciones_list.append(lista[0])
    cv_predicciones_list = np.asarray(cv_predicciones_list)

    #Datos test
    prediccions = regressio.predict(dades_columna_X_test)
    prediccions_list= []
    for lista in prediccions:
        prediccions_list.append(lista[0])
    prediccions_list = np.asarray(prediccions_list)
    df_predicciones = pandas.DataFrame({'OS' : predict_data_y_test.loc[:,"OS"], 'predicció' : prediccions_list})

    r2_coef = r2_score(y_true=predict_data_y_test, y_pred=prediccions)
    print(r2_coef)

    Aic = calculate_aic(modelo = regressio, X  = dades_columna_X_test,y = predict_data_y_test)
    plt.plot(predict_data_y_test, prediccions_list)
    plt.show()
    return r2_coef, Aic, nom_columna
    


print(seleccio_columnes(["CPS","TNM","Okuda","CLIP_score","BCLC"],clinical_data_X_train, clinical_data_X_test))

def regressió_columnes(columnes,clinical_data_X_train,clinical_data_X_test):
    dades_columnes = pandas.DataFrame()
    dades = []
    for nom in columnes:
        for var in clinical_data_X_train.columns:
            if nom in var:
                dades.append(var)
    dades_columna_X_train = clinical_data_X_train.loc[:,dades]
    dades_columna_X_test = clinical_data_X_test.loc[:,dades]
    names_Xtrain= dades_columna_X_train.index.to_list()
    names_Xtest= dades_columna_X_test.index.to_list()                           #Llista amb l'ID del pacients que tenen informacio necessaria
    y_train = predict_feature(clinical_data_path, names_Xtrain)     #OS dels mateixos pacients
    y_test = predict_feature(clinical_data_path, names_Xtest)
    return fer_regressio_una_columna("Escales", dades_columna_X_train, dades_columna_X_test,y_train,y_test)

print(regressió_columnes(["CPS","TNM","Okuda","CLIP_score","BCLC"],clinical_data_X_train, clinical_data_X_test))