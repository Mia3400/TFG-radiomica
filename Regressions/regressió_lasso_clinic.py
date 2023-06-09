import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn import  model_selection
from sklearn.model_selection import train_test_split
import numpy as np
from coeficients import r2,calculate_aic
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import seaborn as sns

def print_table(data):
    print("{:<50s}{:<10s}".format("Name", "Coefficient"))
    for coef, name in data:
        print("{:<50s}{:<10f}".format(name, coef))

#regressió només amb les variables clíniques del pacient
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
# 55 variables 
# Diferenciam entre numèriques i qualitatives, pero hem notat abans que hi ha variables descrites 
#Barplot variables qualitatives:
for col in clinical_data.columns:
    unique_vals = clinical_data[col].unique()
    if len(unique_vals) <= 7:
        clinical_data.loc[:,col] = clinical_data.loc[:,col].astype("category")

fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(15, 8))
axes = axes.flat
columnas_object = clinical_data.select_dtypes(include=['category'])

for i, colum in enumerate(columnas_object):
    columnas_object.loc[:,colum].value_counts().plot.barh(ax = axes[i])
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")

    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribució variables',
             fontsize = 15, fontweight = "bold")



# Binarització de variables qualitatives.
#=====================================================================================================================
# print(clinical_data["hepatitis"].value_counts()) #No virus,HCV only,HCV and HBV , HBV only
dummies_hepatitis = pandas.get_dummies(clinical_data['hepatitis'],prefix="hepatitis").drop(['hepatitis_No virus'], axis=1)

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
dummies_CPS = pandas.get_dummies(clinical_data['CPS'],prefix='CPS')

# print(clinical_data["tumor_nodul"].value_counts())#és binària per tant 
dummies_tumor_nodul = pandas.get_dummies(clinical_data['tumor_nodul'],prefix='tumor_nodul').drop(['tumor_nodul_uninodular'], axis=1)

# print(clinical_data["T_involvment"].value_counts())#<=50%,>50%
encoder = OrdinalEncoder(categories=[["< or = 50%", ">50%"]])
encoder.fit(clinical_data[["T_involvment"]])
clinical_data["T_involvment"] = encoder.transform(clinical_data[["T_involvment"]])

# print(clinical_data["AFP_group"].value_counts())#<400,>=400
encoder = OrdinalEncoder(categories=[["<400", ">=400"]])
encoder.fit(clinical_data[["AFP_group"]])
clinical_data["AFP_group"] = encoder.transform(clinical_data[["AFP_group"]])

# print(clinical_data["CLIP_Score"].value_counts())
dummies_CLIP_Score = pandas.get_dummies(clinical_data['CLIP_Score'],prefix='CLIP_score')

# print(clinical_data["CLIP"].value_counts())
dummies_CLIP = pandas.get_dummies(clinical_data['CLIP'],prefix='CLIP')
# print(clinical_data["Okuda"].value_counts())
dummies_Okuda = pandas.get_dummies(clinical_data['Okuda'],prefix='Okuda')

# print(clinical_data["TNM"].value_counts())
dummies_TNM = pandas.get_dummies(clinical_data['TNM'],prefix= 'TNM')

# print(clinical_data["BCLC"].value_counts())
dummies_BCLC= pandas.get_dummies(clinical_data['BCLC'],prefix='BCLC')

# Afagim les variables qualitatives tractades amb get_dummies i llevam les anteriors
clinical_data = pandas.concat([clinical_data, dummies_hepatitis,dummies_tumor_nodul,dummies_CPS,dummies_CLIP_Score,dummies_TNM,dummies_BCLC,dummies_Okuda, dummies_CLIP], axis = 1)
clinical_data = clinical_data.drop(columns=['tumor_nodul','hepatitis','CPS','CLIP_Score','TNM','BCLC','Okuda','CLIP'])

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
#Ara podem separar per numèriques i categòriques

#VARIABLES NUMÈRIQUES:
#======================================================================================
numeriques = clinical_data.select_dtypes(include=['float64', 'int'])

# print(numeriques.describe())
# De moment mini analisis de correlacions entre elles:
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
plt.close()

#Hi ha variables MOLT correlacionades en aquest cas, es podria descartar features en aquest pas, entenc que Lasso ja ho fa mirant la cantitat de infromacóque ens donen 
#pero no estic segura.

#Separació:
#=============================================================================================
#A la pràctica sempre es sol fer 0.7 i 0.3, però tenc molt poques mostres, no se si es petit encare que després fassem cross validation
X_train, X_test, y_train, y_test = train_test_split(clinical_data, predict_data, train_size = 0.7,shuffle = True, random_state= 1234)

#Processament de les dades:
#=============================================================================================
#Valors NA's:
total = X_train.isnull().sum().sort_values(ascending = False)
percent = (X_train.isnull().sum() / X_train.isnull().count()).sort_values(ascending = False)
missing_data = pandas.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

X_train = X_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

#Estandardització:
#=============================================================================================
X_train.info()
for colname in X_test.columns:
    if colname not in X_train.columns:
        X_test = X_test.drop(colname, axis = 1)
X_train_num = list(X_train.select_dtypes(include=['float64', 'int']).columns)

scaler = StandardScaler()
X_train.loc[:,X_train_num] = scaler.fit_transform(X_train.loc[:,X_train_num])
X_test.loc[:,X_train_num] = scaler.fit_transform(X_test.loc[:,X_train_num])

#Elecció alpha
print('NUMERO VARIABLES FINAL PROCESSAMENT:')
print(X_train.shape)

alphas = np.linspace(0.01,30,200)
lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('Coeficients')
plt.title('Coeficients en funció d\'alpha')
plt.show()
plt.close()
#===================================================
Lasso_reg= Lasso(alpha=0.9, random_state=1234)
Lasso_reg.fit(X_train,y_train)
def print_table(data):
    print("{:<50s}{:<10s}".format("Name", "Coefficient"))
    for coef, name in data:
        print("{:<50s}{:<10f}".format(name, coef))

alpha_coefs = list(zip(Lasso_reg.coef_, X_train))
print_table(alpha_coefs)
#===============================================================================================
#Avaluació del model
# metriques amb training
scoring = "r2"
results = cross_val_score(Lasso_reg, X_train, y_train, cv=4, scoring=scoring)
print(results)
print("R squared training val: ", results.mean())


results = cross_val_score(Lasso_reg, X_train, y_train, cv=4, scoring=r2)
print(results)
print("R squared val: ", results.mean())


#Anàlisi de resiuds:
cv_prediccones = cross_val_predict(
                    estimator = Lasso_reg,
                    X         = X_train,
                    y         = y_train,
                    cv        = 4
                 )

# Gràfics
# ==============================================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))

axes[0, 0].scatter(y_train, cv_prediccones, edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 0].plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    'k--', color = 'black', lw=2
)
axes[0, 0].set_title('Valor predit vs valor real', fontsize = 10, fontweight = "bold")
axes[0, 0].set_xlabel('Real')
axes[0, 0].set_ylabel('Predicció')
axes[0, 0].tick_params(labelsize = 7)

axes[0, 1].scatter(list(range(len(y_train))), y_train.loc[:,"OS"].tolist() - cv_prediccones,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[0, 1].set_title('Residus del model', fontsize = 10, fontweight = "bold")
axes[0, 1].set_xlabel('id')
axes[0, 1].set_ylabel('Residu')
axes[0, 1].tick_params(labelsize = 7)

sns.histplot(
    data    = y_train.loc[:,"OS"].tolist() - cv_prediccones,
    stat    = "density",
    kde     = True,
    line_kws= {'linewidth': 1},
    color   = "firebrick",
    alpha   = 0.3,
    ax      = axes[1, 0]
)

axes[1, 0].set_title('Histograma de residus', fontsize = 10,
                     fontweight = "bold")
axes[1, 0].set_xlabel("residu")
axes[1, 0].set_ylabel("densitat")
axes[1, 0].tick_params(labelsize = 7)


sm.qqplot(
    y_train.loc[:,"OS"].tolist() - cv_prediccones,
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

#Datos test
prediccions = Lasso_reg.predict(X_test)
df_predicciones = pandas.DataFrame({'OS' : y_test.loc[:,"OS"], 'prediccion' : prediccions})
print(df_predicciones)

print("R2 computat per jo a test:")
print(r2(Lasso_reg,X_test,y_test))
r2_score = r2_score(y_true=y_test, y_pred=prediccions)
print(r2_score)
print("AIC")
print(calculate_aic(Lasso_reg,X_test,y_test))