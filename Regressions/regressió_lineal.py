import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn import  model_selection
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import seaborn as sns

def print_table(data):
    print("{:<50s}{:<10s}".format("Name", "Coefficient"))
    for coef, name in data:
        print("{:<50s}{:<10f}".format(name, coef))
#regressió LINEAL només amb les variables clíniques del pacient
#Dades:
#===============================================================================
clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
data = pandas.read_excel(clinical_data_path)


predict = data.loc[:,["TCIA_ID","OS"]]
predict_data = pandas.DataFrame(predict).set_index("TCIA_ID")

clinical_data = data.loc[:, data.columns != 'OS'].set_index("TCIA_ID")
#Anàlisis exploratori:
#================================================================================

clinical_data.info()
print(clinical_data.shape)
#55 variables 
print(clinical_data.isna().sum().sort_values())
#qualqunes tenen NA's


#diferenciam entre numèriques i qualitatives, pero hem notat abans que hi ha variables descrites 
# Binarització de variables qualitatives.
#=====================================================================================================================
# print(clinical_data["hepatitis"].value_counts()) #No virus,HCV only,HCV and HBV , HBV only
dummies_hepatitis = pandas.get_dummies(clinical_data['hepatitis']).drop(['No virus'], axis=1)

# print(clinical_data["agegp"].value_counts()) #>70 , 61-70 , 51-60, 41-50 , <=40
encoder = OrdinalEncoder(categories=[["<=40", "41-50","51-60","61-70",">70"]])
encoder.fit(clinical_data[["agegp"]])
clinical_data["agegp"] = encoder.transform(clinical_data[["agegp"]])

# print(clinical_data["Pathology"].value_counts()) #Well differentiated,Moderately differentiated,NOT STATED,Poorly differentiated,No biopsy,Well-moderately differentiated
clinical_data['Pathology'] = clinical_data['Pathology'].replace(['NOT STATED', 'No biopsy'], 'No information')     
encoder = OrdinalEncoder(categories=[["No information","Poorly differentiated", "Moderately-poorly differentiated","Moderately differentiated","Well-moderately differentiated","Well differentiated"]])
encoder.fit(clinical_data[["Pathology"]])
clinical_data["Pathology"] = encoder.transform(clinical_data[["Pathology"]])

# print(clinical_data["CPS"].value_counts()) # A,B,C es severitat están ordenats
dummies_CPS = pandas.get_dummies(clinical_data['CPS'])

# print(clinical_data["tumor_nodul"].value_counts())#és binària per tant 
dummies_tumor_nodul = pandas.get_dummies(clinical_data['tumor_nodul']).drop(['uninodular'], axis=1)

# print(clinical_data["T_involvment"].value_counts())#<=50%,>50%
encoder = OrdinalEncoder(categories=[["< or = 50%", ">50%"]])
encoder.fit(clinical_data[["T_involvment"]])
clinical_data["T_involvment"] = encoder.transform(clinical_data[["T_involvment"]])

# print(clinical_data["AFP_group"].value_counts())#<400,>=400
encoder = OrdinalEncoder(categories=[["<400", ">=400"]])
encoder.fit(clinical_data[["AFP_group"]])
clinical_data["AFP_group"] = encoder.transform(clinical_data[["AFP_group"]])

# print(clinical_data["CLIP_Score"].value_counts())
dummies_CLIP_Score = pandas.get_dummies(clinical_data['CLIP_Score'])

# print(clinical_data["CLIP"].value_counts())
encoder = OrdinalEncoder(categories=[["Stage_ 0-2","Stage_3","Stage_4-6"]])
encoder.fit(clinical_data[["CLIP"]])
clinical_data["CLIP"] = encoder.transform(clinical_data[["CLIP"]])

# print(clinical_data["Okuda"].value_counts())
encoder = OrdinalEncoder(categories=[["Stage I","Stage II","Stage III"]])
encoder.fit(clinical_data[["Okuda"]])
clinical_data["Okuda"] = encoder.transform(clinical_data[["Okuda"]])

# print(clinical_data["TNM"].value_counts())
dummies_TNM = pandas.get_dummies(clinical_data['TNM'])

# print(clinical_data["BCLC"].value_counts())
encoder = OrdinalEncoder(categories=[["Stage-A","Stage-B","Stage-C","Stage-D"]])
encoder.fit(clinical_data[["BCLC"]])
clinical_data["BCLC"] = encoder.transform(clinical_data[["BCLC"]])

# Afagim les variables qualitatives tractades amb get_dummies i llevam les anteriors
clinical_data = pandas.concat([clinical_data, dummies_hepatitis,dummies_tumor_nodul,dummies_CPS,dummies_CLIP_Score,dummies_TNM], axis = 1)
clinical_data = clinical_data.drop(columns=['tumor_nodul','hepatitis','CPS','CLIP_Score','TNM'])

#Eliminam les variables on un factor es massa dominant:
#Nose si te molt de sentit incloure la variable de: Death_stillAlive_orlosttoFU...l'elimin de moment
clinical_data = clinical_data.drop(["Metastasis","fhx_livc","Death_1_StillAliveorLostToFU_0"],axis = 1)

#=======================================================================================================================
# Hem de canviar el tipus de variable si están expressades amb nombres però son facotrs
#Mirant les dades he vost que n'hi ha amb fins a 5 factors, les transformam a "category" que es una espècie de "factor" de R.
for col in clinical_data.columns:
    unique_vals = clinical_data[col].unique()
    if len(unique_vals) <= 7:
        clinical_data.loc[:,col] = clinical_data.loc[:,col].astype("category")
#Nose si te molt de sentit incloure la variable de: Death_stillAlive_orlosttoFU...l'elimin de moment
clinical_data_csv = clinical_data.to_csv("clinical_data_processed.csv") 
clinical_data.info()
#Ara podem separar per numèriques i categòriques
#VARIABLES NUMÈRIQUES:
numeriques = clinical_data.select_dtypes(include=['float64', 'int'])

print(numeriques.describe())
#CORRELACIÓ
def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matrix de correlación de pandas en formato tidy
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

corr_matrix = clinical_data.select_dtypes(include=['float64', 'int']).corr(method='pearson')
tidy_corr_matrix(corr_matrix).head(10)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

sns.heatmap(corr_matrix, annot = True, cbar = False,annot_kws = {"size": 6},vmin= -1,vmax = 1,center = 0,cmap = sns.diverging_palette(20, 220, n=200),square = True,ax = ax)
ax.set_xticklabels(ax.get_xticklabels(),rotation = 45, horizontalalignment = 'right')
ax.tick_params(labelsize = 8)
plt.show()
#Hi ha variables MOLT correlacionades en aquest cas, es podria descartar features en aquest pas.

#Separació:
X_train, X_test, y_train, y_test = train_test_split(clinical_data, predict_data, train_size = 0.7,shuffle = True, random_state= 1234)

#Processament de les dades:

#Valors NA's:
total = X_train.isnull().sum().sort_values(ascending = False)
percent = (X_train.isnull().sum() / X_train.isnull().count()).sort_values(ascending = False)
missing_data = pandas.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
print(missing_data)
#Tumor size: 51% de valors nuls...
#Chemoterapy : 15% nuls
#anem a observar come s la variable de quimioterapia:*--

X_train = X_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
print(X_train.isnull().sum().max()) # Para comprobar que no hay más datos desaparecidos.
print(X_train.shape)

#Estandardització
X_train.info()
for colname in X_test.columns:
    if colname not in X_train.columns:
        X_test = X_test.drop(colname, axis = 1)
X_train_num = list(X_train.select_dtypes(include=['float64', 'int']).columns)


scaler = StandardScaler()
X_train.loc[:,X_train_num] = scaler.fit_transform(X_train.loc[:,X_train_num])
X_test.loc[:,X_train_num] = scaler.fit_transform(X_test.loc[:,X_train_num])

regressio= LinearRegression()
regressio.fit(X_train,y_train)
#===============================================================================================
#Avaluació del model
# metriques amb training
scoring = "neg_mean_squared_error"
results = cross_val_score(regressio, X_train, y_train, cv=4, scoring=scoring)
print("Mean Squared Error: ", results.mean())

scoring = "r2"
results = cross_val_score(regressio, X_train, y_train, cv=4, scoring=scoring)
print(results)
print("R squared val: ", results.mean())
print(r2_score(y_pred=regressio.predict(X_train), y_true =  y_train))

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
axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
axes[0, 0].set_xlabel('Real')
axes[0, 0].set_ylabel('Predicción')
axes[0, 0].tick_params(labelsize = 7)

axes[0, 1].scatter(list(range(len(y_train))), y_train.loc[:,"OS"].tolist() - cv_predicciones_list,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
axes[0, 1].set_xlabel('id')
axes[0, 1].set_ylabel('Residuo')
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

axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10,
                     fontweight = "bold")
axes[1, 0].set_xlabel("Residuo")
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
axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
axes[1, 1].tick_params(labelsize = 7)

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold");
plt.show()

#Datos test
prediccions = regressio.predict(X_test)
prediccions_list= []
for lista in prediccions:
    prediccions_list.append(lista[0])
prediccions_list = np.asarray(prediccions_list)
df_predicciones = pandas.DataFrame({'OS' : y_test.loc[:,"OS"], 'prediccion' : prediccions_list})

mse = mean_squared_error(y_true = y_test,y_pred = prediccions)
print(mse)
r2 = r2_score(y_true=y_test, y_pred=prediccions)
print(r2)

