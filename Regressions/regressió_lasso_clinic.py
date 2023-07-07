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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def print_table(data):
    print("{:<50s}{:<10s}".format("Name", "Coefficient"))
    for coef, name in data:
        print("{:<50s}{:<10f}".format(name, coef))

# Regressió amb les variables clíniques del pacient
# Extracció de les dades:
#===============================================================================
clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
data = pandas.read_excel(clinical_data_path)


predict = data.loc[:,["TCIA_ID","OS"]]
predict_data = pandas.DataFrame(predict).set_index("TCIA_ID")

clinical_data = data.loc[:, data.columns != 'OS'].set_index("TCIA_ID")
# Anàlisis exploratori:
#================================================================================

clinical_data.info()
# Barplot variables qualitatives:
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
#========================================================
dummies_hepatitis = pandas.get_dummies(clinical_data['hepatitis'],prefix="hepatitis").drop(['hepatitis_No virus'], axis=1)

encoder = OrdinalEncoder(categories=[["<=40", "41-50","51-60","61-70",">70"]])
encoder.fit(clinical_data[["agegp"]])
clinical_data["agegp"] = encoder.transform(clinical_data[["agegp"]])

clinical_data['Pathology'] = clinical_data['Pathology'].replace(['NOT STATED', 'No biopsy'], 'No information')     
encoder = OrdinalEncoder(categories=[["No information","Poorly differentiated", "Moderately-poorly differentiated","Moderately differentiated","Well-moderately differentiated","Well differentiated"]])
encoder.fit(clinical_data[["Pathology"]])
clinical_data["Pathology"] = encoder.transform(clinical_data[["Pathology"]])

dummies_CPS = pandas.get_dummies(clinical_data['CPS'],prefix='CPS')

dummies_tumor_nodul = pandas.get_dummies(clinical_data['tumor_nodul'],prefix='tumor_nodul').drop(['tumor_nodul_uninodular'], axis=1)

encoder = OrdinalEncoder(categories=[["< or = 50%", ">50%"]])
encoder.fit(clinical_data[["T_involvment"]])
clinical_data["T_involvment"] = encoder.transform(clinical_data[["T_involvment"]])

encoder = OrdinalEncoder(categories=[["<400", ">=400"]])
encoder.fit(clinical_data[["AFP_group"]])
clinical_data["AFP_group"] = encoder.transform(clinical_data[["AFP_group"]])

dummies_CLIP_Score = pandas.get_dummies(clinical_data['CLIP_Score'],prefix='CLIP_score')

dummies_CLIP = pandas.get_dummies(clinical_data['CLIP'],prefix='CLIP')

dummies_Okuda = pandas.get_dummies(clinical_data['Okuda'],prefix='Okuda')

dummies_TNM = pandas.get_dummies(clinical_data['TNM'],prefix= 'TNM')

dummies_BCLC= pandas.get_dummies(clinical_data['BCLC'],prefix='BCLC')

# Afagim les variables qualitatives tractades amb get_dummies i llevam les anteriors:
clinical_data = pandas.concat([clinical_data, dummies_hepatitis,dummies_tumor_nodul,dummies_CPS,dummies_CLIP_Score,dummies_TNM,dummies_BCLC,dummies_Okuda, dummies_CLIP], axis = 1)
clinical_data = clinical_data.drop(columns=['tumor_nodul','hepatitis','CPS','CLIP_Score','TNM','BCLC','Okuda','CLIP'])

#Eliminam les variables on un factor es massa dominant // no tenen sentit a la regressió:
clinical_data = clinical_data.drop(["Metastasis","fhx_livc","Death_1_StillAliveorLostToFU_0"],axis = 1)

#Transformam les variables amb menys de 7 factors únics en variables categòriques:
for col in clinical_data.columns:
    unique_vals = clinical_data[col].unique()
    if len(unique_vals) <= 7:
        clinical_data.loc[:,col] = clinical_data.loc[:,col].astype("category")

# Les varaibels numèriques:
#======================================================================================
numeriques = clinical_data.select_dtypes(include=['float64', 'int'])

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

clinical_data_bin = clinical_data.to_csv("clinical_data_bin.csv")
#Separació entrenament i test:
#=============================================================================================
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
for colname in X_test.columns:
    if colname not in X_train.columns:
        X_test = X_test.drop(colname, axis = 1)

clinical_data_train = X_train.to_csv("clinical_data_Xtrain_processed.csv") 
clinical_data_test = X_test.to_csv("clinical_data_Xtest_processed.csv") 
clinical_data_train = y_train.to_csv("clinical_data_Ytrain_processed.csv") 
clinical_data_test = y_test.to_csv("clinical_data_Ytest_processed.csv") 

X_train_num = list(X_train.select_dtypes(include=['float64', 'int']).columns)
scaler = StandardScaler()
X_train.loc[:,X_train_num] = scaler.fit_transform(X_train.loc[:,X_train_num])
X_test.loc[:,X_train_num] = scaler.fit_transform(X_test.loc[:,X_train_num])

#Elecció hiperparàmetre alpha mitjançant cross-validation:

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

#Cross-validation
#===============================================================================================
alphas_propostes = np.linspace(0.01,1,20)
Lassoreg= LassoCV(alphas= alphas_propostes,cv = 4,random_state=1234).fit(X_train,y_train)
print("L'hiperparàmetre elegit és" , Lassoreg.alpha_)

#Model final
#===============================================================================================
Lasso_reg= Lasso(alpha=1, random_state=1234)
Lasso_reg.fit(X_train,y_train)
def print_table(data):
    print("{:<50s}{:<10s}".format("Variable", "Coeficient"))
    for coef, name in data:
        print("{:<50s}{:<10f}".format(name, coef))

alpha_coefs = list(zip(Lasso_reg.coef_, X_train))
print_table(alpha_coefs)
#Avaluació del model
#===============================================================================================
# Subconjunt de training
scoring = "r2"
results = cross_val_score(Lasso_reg, X_train, y_train, cv=4, scoring=scoring)
print(results)
print("R squared training val: ", results.mean())

#Anàlisi de resiuds:
cv_prediccones = cross_val_predict(
                    estimator = Lasso_reg,
                    X         = X_train,
                    y         = y_train,
                    cv        = 4
                 )

# Gràfics
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 5))

axes[0, 0].scatter(y_train, cv_prediccones, edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 0].plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    'k--',  lw=2
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

# Avaluació a les dades test
prediccions = Lasso_reg.predict(X_test)
df_predicciones = pandas.DataFrame({'OS' : y_test.loc[:,"OS"], 'predicció' : prediccions})
print(df_predicciones)

r2_valor= r2_score(y_true=y_test, y_pred=prediccions)
print("R2 computat per la nostra funció al subconjunt de test ", r2_valor)
print("L'AIC ve donat per")
print(calculate_aic(Lasso_reg,X_test,y_test))

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