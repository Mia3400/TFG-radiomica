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
#A aquest document es realitzarán regressions amb característiques clíniques que destaquen mèdicamtn per intentar millorar el model fet
clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
clinical_data_processed = pandas.read_csv("clinical_data_processed.csv").set_index("TCIA_ID")
clinical_data = pandas.read_excel(clinical_data_path)
predict_data = clinical_data.loc[:,["OS", "TCIA_ID"]].set_index("TCIA_ID")

def seleccio_columnes(columnes, dades_cliniques): 
    taula = []
    dades_columna = pandas.DataFrame()
    for nom in columnes:
        dades = []
        for var in dades_cliniques.columns:
            if nom in var:
                dades.append(var)
        dades_columna = dades_cliniques.loc[:,dades]
        r2_score , Aic , nom_columna = fer_regressio_una_columna(nom,dades_columna,predict_data)
        taula.append({
            'Columna': nom_columna,
            'r2_score': r2_score,
            'Aic': Aic
        })
    df = pandas.DataFrame(taula)
    return df.set_index("Columna")


def fer_regressio_una_columna(nom_columna,dades_columna,predict_data):
    X_train, X_test, y_train, y_test = train_test_split(dades_columna, predict_data, train_size = 0.7,shuffle = True, random_state= 1234)
    regressio= LinearRegression()
    regressio.fit(X_train,y_train)
    scoring = "r2"
    results = cross_val_score(regressio, X_train, y_train, cv=4, scoring=scoring)
    print(results)
    print("R squared val ", nom_columna, results.mean())

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

    # # Gràfics
    # # ==============================================================================
    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))

    # axes[0, 0].scatter(y_train, cv_predicciones, edgecolors=(0, 0, 0), alpha = 0.4)
    # axes[0, 0].plot(
    #     [y_train.min(), y_train.max()],
    #     [y_train.min(), y_train.max()],
    #     'k--', color = 'black', lw=2
    # )
    # axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
    # axes[0, 0].set_xlabel('Real')
    # axes[0, 0].set_ylabel('Predicción')
    # axes[0, 0].tick_params(labelsize = 7)

    # axes[0, 1].scatter(list(range(len(y_train))), y_train.loc[:,"OS"].tolist() - cv_predicciones_list,
    #                 edgecolors=(0, 0, 0), alpha = 0.4)
    # axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    # axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
    # axes[0, 1].set_xlabel('id')
    # axes[0, 1].set_ylabel('Residuo')
    # axes[0, 1].tick_params(labelsize = 7)

    # sns.histplot(
    #     data    = y_train.loc[:,"OS"].tolist() - cv_predicciones_list,
    #     stat    = "density",
    #     kde     = True,
    #     line_kws= {'linewidth': 1},
    #     color   = "firebrick",
    #     alpha   = 0.3,
    #     ax      = axes[1, 0]
    # )

    # axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10,
    #                     fontweight = "bold")
    # axes[1, 0].set_xlabel("Residuo")
    # axes[1, 0].tick_params(labelsize = 7)


    # sm.qqplot(
    #     y_train.loc[:,"OS"].tolist() - cv_predicciones_list,
    #     fit   = True,
    #     line  = 'q',
    #     ax    = axes[1, 1], 
    #     color = 'firebrick',
    #     alpha = 0.4,
    #     lw    = 2
    # )
    # axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
    # axes[1, 1].tick_params(labelsize = 7)

    # fig.tight_layout()
    # plt.subplots_adjust(top=0.9)
    # fig.suptitle('Diagnòstic de residus '+ nom_columna, fontsize = 12, fontweight = "bold")
    # #plt.show()

    #Datos test
    prediccions = regressio.predict(X_test)
    prediccions_list= []
    for lista in prediccions:
        prediccions_list.append(lista[0])
    prediccions_list = np.asarray(prediccions_list)
    df_predicciones = pandas.DataFrame({'OS' : y_test.loc[:,"OS"], 'predicció' : prediccions_list})
    print(df_predicciones)
    print("Evaluaciones dados test para ", nom_columna)

    print("r2", nom_columna)
    r2_coef = r2_score(y_true=y_test, y_pred=prediccions)
    print(r2_coef)

    print("AIC", nom_columna)
    Aic = calculate_aic(modelo = regressio, X  = X_test,y = y_test)
    print(Aic)
    return r2_coef, Aic, nom_columna
    


# print(seleccio_columnes(["CPS","TNM","Okuda","CLIP_score"],clinical_data_processed))

def regressió_columnes(columnes,dades_cliniques):
    dades_columnes = pandas.DataFrame()
    dades = []
    for nom in columnes:
        for var in dades_cliniques.columns:
            if nom in var:
                dades.append(var)
    dades_columnes = dades_cliniques.loc[:,dades]
    print(dades_columnes.shape)
    return fer_regressio_una_columna("Escales", dades_columnes, predict_data)

print(regressió_columnes(["CPS","TNM","Okuda","CLIP_score"],clinical_data_processed))