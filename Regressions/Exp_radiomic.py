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
        r_valid ,r2_score , Aic  = fer_regressio(dades_grup_X_train,dades_grup_X_test,y_train,y_test)
        taula.append({
                'Columna': grup,
                'r_valid': r_valid,
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
    #dades training:
    cv_predicciones = cross_val_predict(
                        estimator = Lasso_reg,
                        X         = dades_grup_X_train,
                        y         = predict_data_y_train,
                        cv        = 4
                    )
    scoring = "r2"
    results = cross_val_score(Lasso_reg, dades_grup_X_train, predict_data_y_train, cv=4, scoring=scoring)
    print(results)
    r_valid =  results.mean()
        #Grafics de residus
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))

    # axes[0, 0].scatter(predict_data_y_train, cv_predicciones, edgecolors=(0, 0, 0), alpha = 0.4)
    # axes[0, 0].plot(
    #     [predict_data_y_train.min(), predict_data_y_train.max()],
    #     [predict_data_y_train.min(), predict_data_y_train.max()],
    #     'k--', color = 'black', lw=2
    # )
    # axes[0, 0].set_title('Valor predit vs valor real', fontsize = 10, fontweight = "bold")
    # axes[0, 0].set_xlabel('Real')
    # axes[0, 0].set_ylabel('Predicció')
    # axes[0, 0].tick_params(labelsize = 7)

    # axes[0, 1].scatter(list(range(len(predict_data_y_train))), predict_data_y_train.loc[:,"OS"].tolist() - cv_predicciones,
    #                 edgecolors=(0, 0, 0), alpha = 0.4)
    # axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    # axes[0, 1].set_title('Residus del model', fontsize = 10, fontweight = "bold")
    # axes[0, 1].set_xlabel('id')
    # axes[0, 1].set_ylabel('Residu')
    # axes[0, 1].tick_params(labelsize = 7)

    # sns.histplot(
    #     data    = predict_data_y_train.loc[:,"OS"].tolist() - cv_predicciones,
    #     stat    = "density",
    #     kde     = True,
    #     line_kws= {'linewidth': 1},
    #     color   = "firebrick",
    #     alpha   = 0.3,
    #     ax      = axes[1, 0]
    # )

    # axes[1, 0].set_title('Distribució residus del model', fontsize = 10,
    #                     fontweight = "bold")

    # axes[1, 0].set_xlabel("Residu")
    # axes[1, 0].set_ylabel("densitat")
    # axes[1, 0].tick_params(labelsize = 7)


    # sm.qqplot(
    #     predict_data_y_train.loc[:,"OS"].tolist() - cv_predicciones,
    #     fit   = True,
    #     line  = 'q',
    #     ax    = axes[1, 1], 
    #     color = 'firebrick',
    #     alpha = 0.4,
    #     lw    = 2
    # )
    # axes[1, 1].set_title('Q-Q residus del model', fontsize = 10, fontweight = "bold")
    # axes[1, 1].set_xlabel("Quantils teòrics")
    # axes[1, 1].set_ylabel("Quantils mostrals")
    # axes[1, 1].tick_params(labelsize = 7)

    # fig.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # fig.suptitle('Diagnòstic de residus', fontsize = 12, fontweight = "bold")
    # plt.show()

    print(counts)
    prediccions = Lasso_reg.predict(dades_grup_X_test)
    df_predicciones = pandas.DataFrame({'OS' : predict_data_y_test.loc[:,"OS"], 'predicció' : prediccions})
    #print(df_predicciones)

    r2_coef = r2_score(y_true= predict_data_y_test,y_pred= prediccions)
    print(r2_coef)
    print("AIC")
    Aic = calculate_aic(Lasso_reg,dades_grup_X_test,predict_data_y_test)
    print(Aic)
    return r_valid,r2_coef,Aic

print(regressio_grup_radiomic(["firstorder","glcm","shape","glszm","glrlm","gldm","ngtdm"], clinical_data_X_test= clinical_data_X_test,clinical_data_X_train=clinical_data_X_train, radiomic_data=caract_radiomiques))




