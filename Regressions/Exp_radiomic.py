#A aquest document realitzarem regressions amb les característiques radiòmiques de cada grup
import pandas 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from coeficients import r2,calculate_aic
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale 

def predict_feature(path_clinical_data, names):
    predict = []
    clinical_data = pandas.read_excel(path_clinical_data)
    for i in range (0,len(clinical_data)):
        if clinical_data.loc[i,"TCIA_ID"] in names:
            predict.append(clinical_data.loc[i,["TCIA_ID","OS"]])
    predict_data = pandas.DataFrame(predict).set_index("TCIA_ID")
    return predict_data


clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
clinical_data = pandas.read_excel(clinical_data_path)

clinical_data_processed = pandas.read_csv("clinical_data_processed.csv")
clinical_data_processed = clinical_data_processed.rename(columns={"TCIA_ID":"Patient"})


caract_radiomiques = pandas.read_csv("feature_data_2.csv")



def regressio_grup_radiomic(grup, clinical_data, radiomic_data):
    var_grup = ["Patient"]
    taula = []
    for var in radiomic_data.columns:
        if grup in var:
            var_grup.append(var)
    dades_grup = pandas.DataFrame(radiomic_data.loc[:,var_grup])
    dades_grup_cliniques = pandas.merge(dades_grup,clinical_data, on="Patient").set_index("Patient")
    names = dades_grup_cliniques.index.to_list()
    predict_data = predict_feature(clinical_data_path, names)
    r2_score , Aic  = fer_regressio(dades_grup_cliniques,predict_data)
    taula.append({
            'Columna': grup,
            'r2_score': r2_score,
            'Aic': Aic
        })
    df = pandas.DataFrame(taula)
    return df.set_index("Columna")

def fer_regressio(dades, predict_data):
    for col in dades.columns:
        unique_vals = dades[col].unique()
        if len(unique_vals) <= 7:
            dades.loc[:,col] = dades.loc[:,col].astype("category")
    X_train, X_test, y_train, y_test = train_test_split(dades, predict_data, train_size   = 0.7,shuffle = True, random_state= 1234)
    X_train = X_train.drop(X_train.columns[X_train.isna().any()],1)
    for colname in X_test.columns:
        if colname not in X_train.columns:
            X_test = X_test.drop(colname, axis = 1)
    X_train_num = X_train.select_dtypes(include=['float64', 'int'])
    col_num = X_train_num.columns
    scaler = StandardScaler()
    X_train.loc[:,col_num] = scaler.fit_transform(X_train.loc[:,col_num])
    X_test.loc[:,col_num] = scaler.fit_transform(X_test.loc[:,col_num])
    Lasso_reg= Lasso(alpha=0.3, random_state=42)
    Lasso_reg.fit(X_train,y_train)
# scores = cross_val_score(Lasso_reg, X_train, y_train_n, cv=3)
# print(scores) #perque son tant dolentes?????

    def print_table(data):
        print("{:<50s}{:<10s}".format("Name", "Coefficient"))
        for coef, name in data:
            print("{:<50s}{:<10f}".format(name, coef))

    alpha_coefs = list(zip(Lasso_reg.coef_, X_train))

    counts = {'original': 0, 'modified': 0}

    for name, coef in alpha_coefs:
        if 'original' in coef and name != 0:
            counts['original'] += 1
        elif 'original' not in coef and name != 0:
            counts['modified'] += 1

    print(counts)
    prediccions = Lasso_reg.predict(X_test)
    df_predicciones = pandas.DataFrame({'OS' : y_test.loc[:,"OS"], 'predicció' : prediccions})
    print(df_predicciones)
    r2_coef = r2_score(y_true= y_test,y_pred= prediccions)
    print(r2_coef)
    print("AIC")
    Aic = calculate_aic(Lasso_reg,X_test,y_test)
    print(Aic)
    return r2_coef,Aic

print(regressio_grup_radiomic("firstorder", clinical_data_processed,caract_radiomiques))




