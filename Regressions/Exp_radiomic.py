#A aquest document realitzarem regressions amb les característiques radiòmiques de cada grup
import pandas 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,LassoCV
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

# Extracció de les dades:
clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
clinical_data = pandas.read_excel(clinical_data_path)

clinical_data_processed = pandas.read_csv("clinical_data_processed.csv")
clinical_data_processed = clinical_data_processed.rename(columns={"TCIA_ID":"Patient"})
clinical_data_X_train = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Xtrain_processed.csv").rename(columns={"TCIA_ID":"Patient"})
clinical_data_X_test = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Xtest_processed.csv").rename(columns={"TCIA_ID":"Patient"})
clinical_data_y_train = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Ytrain_processed.csv").rename(columns={"TCIA_ID":"Patient"})
clinical_data_y_test = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Ytest_processed.csv").rename(columns={"TCIA_ID":"Patient"})


caract_radiomiques = pandas.read_csv("feature_data_2.csv")

# Funció que reprodueix el mateix procés que a la regressió original només amb un grup de característiques donades
def regressio_grup_radiomic(grups, clinical_data_X_train,clinical_data_X_test,radiomic_data):
    var_grup = ["Patient"]
    taula = []
    for grup in grups:
        for var in radiomic_data.columns:
            if grup in var:
                var_grup.append(var)
        dades_grup = pandas.DataFrame(radiomic_data.loc[:,var_grup])
        dades_grup_X_test = pandas.merge(dades_grup,clinical_data_X_test, on="Patient").set_index("Patient")
        dades_grup_X_train = pandas.merge(dades_grup,clinical_data_X_train, on="Patient").set_index("Patient")
        names_Xtrain= dades_grup_X_train.index.to_list()
        names_Xtest= dades_grup_X_test.index.to_list()                           #Llista amb l'ID del pacients que tenen informacio necessaria
        y_train = predict_feature(clinical_data_path, names_Xtrain)     #OS dels mateixos pacients
        y_test = predict_feature(clinical_data_path, names_Xtest)
        r2_score , Aic  = fer_regressio(dades_grup_X_train,dades_grup_X_test,y_train,y_test)
        taula.append({
                'Columna': grup,
                'r2_score': r2_score,
                'Aic': Aic
            })
    df = pandas.DataFrame(taula)
    return df.set_index("Columna")

def fer_regressio(dades_grup_X_train, dades_grup_X_test,predict_data_y_train,predict_data_y_test):
    for col in dades_grup_X_train.columns:
        unique_vals = dades_grup_X_train[col].unique()
        if len(unique_vals) <= 7:
            dades_grup_X_train.loc[:,col] = dades_grup_X_train.loc[:,col].astype("category")
    X_train_num = dades_grup_X_train.select_dtypes(include=['float64', 'int'])
    col_num = X_train_num.columns
    if len(col_num)!=0:
        scaler = StandardScaler()
        dades_grup_X_train.loc[:,col_num] = scaler.fit_transform(dades_grup_X_train.loc[:,col_num])
        dades_grup_X_test.loc[:,col_num] = scaler.fit_transform(dades_grup_X_test.loc[:,col_num])
    #elecció de l'alpha mitjançant cross-validation
    alphas_propostes = np.linspace(0.01,0.3,20)
    Lassoreg= LassoCV(alphas= alphas_propostes,cv = 4,random_state=1234).fit(dades_grup_X_train,predict_data_y_train)
    print("L'hiperparàmetre elegit és" , Lassoreg.alpha_)
    Lasso_reg= Lasso(alpha=Lassoreg.alpha_, random_state=1234)
    Lasso_reg.fit(dades_grup_X_train,predict_data_y_train)

    def print_table(data):
        print("{:<50s}{:<10s}".format("Name", "Coefficient"))
        for coef, name in data:
            print("{:<50s}{:<10f}".format(name, coef))

    alpha_coefs = list(zip(Lasso_reg.coef_, dades_grup_X_train))

    counts = {'radiomiques': 0, 'cliniques': 0}

    for name, coef in alpha_coefs:
        if 'original' in coef and name != 0:
            counts['radiomiques'] += 1
        elif 'original' not in coef and name != 0:
            counts['cliniques'] += 1

    print(counts)
    prediccions = Lasso_reg.predict(dades_grup_X_test)
    df_predicciones = pandas.DataFrame({'OS' : predict_data_y_test.loc[:,"OS"], 'predicció' : prediccions})
    #print(df_predicciones)
    r2_coef = r2_score(y_true= predict_data_y_test,y_pred= prediccions)
    print(r2_coef)
    print("AIC")
    Aic = calculate_aic(Lasso_reg,dades_grup_X_test,predict_data_y_test)
    print(Aic)
    return r2_coef,Aic

print(regressio_grup_radiomic(["firstorder","glcm","shape","glszm","glrlm","gldm","ngtdm"], clinical_data_X_test= clinical_data_X_test,clinical_data_X_train=clinical_data_X_train, radiomic_data=caract_radiomiques))




