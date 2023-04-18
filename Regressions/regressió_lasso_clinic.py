import pandas
import matplotlib.pyplot as plt
from group_lasso import GroupLasso
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import seaborn as sns

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
print(clinical_data.shape)
#55 variables 
print(clinical_data.isna().sum().sort_values())
#qualqunes tenen NA's

distribucion_variable = sns.kdeplot(predict_data.loc[:,"OS"], fill    = True, color   = "blue")

#diferenciam entre numèriques i qualitatives, pero hem notat abans que hi ha variables descrites 

# Hem de canviar el tipus de variable si están expressades amb nombres però son facotrs
#Mirant les dades he vost que n'hi ha amb fins a 5 factors, les transformam a "category" que es una espècie de "factor" de R.
for col in clinical_data.columns:
    unique_vals = clinical_data[col].unique()
    if len(unique_vals) <= 5:
        clinical_data.loc[:,col] = clinical_data.loc[:,col].astype("category")

clinical_data.info()
#Ara podem separar per numèriques i categòriques
#VARIABLES NUMÈRIQUES:
numeriques = clinical_data.select_dtypes(include=['float64', 'int'])

print(numeriques.describe())
#De moment mini analisis de correlacions entre elles:
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
#Hi ha variables MOLT correlacionades en aquest cas, es podria descartar features en aquest pas, entenc que Lasso ja ho fa mirant la cantitat de infromacóque ens donen 
#pero no estic segura.

#Variables qualitatives:
#================================================
qualitatives = clinical_data.select_dtypes(include=["object", "category"])
print(qualitatives.describe())
#Problema: me preocupa que qualquna d'aquestes variables tengui molts pocs valors a certes caract i efecti a sa cross-validation si a un grup des k 
# per ser tant pocs no n'agafa.
fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(20,10))
axes = axes.flat

for i, colum in enumerate(qualitatives):
    clinical_data[colum].value_counts().plot.barh(ax = axes[i])
    axes[i].set_title(colum, fontsize = 7, fontweight = "bold")
    axes[i].tick_params(labelsize = 6)
    axes[i].set_xlabel("")

    
fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Distribución variables cualitativas',
             fontsize = 10, fontweight = "bold")


#Separació:
#A la pràctica sempre es sol fer 0.8 i 0.2, però tenc molt poques mostres, no se si es petit encare que després fassem cross validation
X_train, X_test, y_train, y_test = train_test_split(clinical_data, predict_data, train_size   = 0.8,shuffle = True, random_state= 1234)

#Pre-processament de les dades:
#Valors NA's:
total = X_train.isnull().sum().sort_values(ascending = False)
percent = (X_train.isnull().sum() / X_train.isnull().count()).sort_values(ascending = False)
missing_data = pandas.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
print(missing_data)
#Tumor size: 51% de valors nuls...
#Chemoterapy : 15% nuls
#anem a observar come s la variable de quimioterapia:*--
#no tenc clar que fer amb aquests nuls, crec que recordant erl que vas dir imputar-los computacionalment amb tant poques dades 
# no es gaire natural i la regressió es veura afectada, a més les quetenen valors nuls están MOLT correlacionades aií que de moment les he eliminades 

X_train = X_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
print(X_train.isnull().sum().max()) # Para comprobar que no hay más datos desaparecidos.
print(X_train.shape)
#Outliers:
boxplot_y = plt.boxplot(y_train)

#Estandardització
X_train.info()
X_train_num = X_train.select_dtypes(include=['float64', 'int'])
X_train_num.info()

scaler = StandardScaler()
X_train_n = scaler.fit_transform(X_train_num)
y_train_n = scaler.fit_transform(y_train)
plt.boxplot(y_train_n)
plt.show()

# Binarització variables qualitatives: ho he fet al principi pel problema explicat al latex
# Binarització de variables qualitatives.
#=====================================================================================================================
print(X_train["hepatitis"].value_counts()) #No virus,HCV only,HCV and HBV , HBV only
dummies_hepatitis = pandas.get_dummies(X_train['hepatitis']).drop(['No virus'], axis=1)

print(X_train["agegp"].value_counts()) #>70 , 61-70 , 51-60, 41-50 , <=40
encoder = OrdinalEncoder(categories=[["<=40", "41-50","51-60","61-70",">70"]])
encoder.fit(X_train[["agegp"]])
X_train["agegp"] = encoder.transform(X_train[["agegp"]])

print(X_train["Pathology"].value_counts()) #Well differentiated,Moderately differentiated,NOT STATED,Poorly differentiated,No biopsy,Well-moderately differentiated     
dummies_pathology = pandas.get_dummies(X_train['Pathology']).drop(['No biopsy'], axis=1)

print(X_train["CPS"].value_counts()) # A,B,C es severitat están ordenats
encoder = OrdinalEncoder(categories=[["A", "B","C"]])
encoder.fit(X_train[["CPS"]])
X_train["CPS"] = encoder.transform(X_train[["CPS"]])

print(X_train["tumor_nodul"].value_counts())#és binària per tant 
dummies_tumor_nodul = pandas.get_dummies(X_train['tumor_nodul']).drop(['uninodular'], axis=1)

print(X_train["T_involvment"].value_counts())#<=50%,>50%
encoder = OrdinalEncoder(categories=[["< or = 50%", ">50%"]])
encoder.fit(X_train[["T_involvment"]])
X_train["T_involvment"] = encoder.transform(X_train[["T_involvment"]])

print(X_train["AFP_group"].value_counts())#<400,>=400
encoder = OrdinalEncoder(categories=[["<400", ">=400"]])
encoder.fit(X_train[["AFP_group"]])
X_train["AFP_group"] = encoder.transform(X_train[["AFP_group"]])

print(X_train["CLIP_Score"].value_counts())
encoder = OrdinalEncoder(categories=[["Stage_0","Stage_1","Stage_2","Stage_3","Stage_4","Stage_5"]])
encoder.fit(X_train[["CLIP_Score"]])
X_train["CLIP_Score"] = encoder.transform(X_train[["CLIP_Score"]])

print(X_train["CLIP"].value_counts())
encoder = OrdinalEncoder(categories=[["Stage_ 0-2","Stage_3","Stage_4-6"]])
encoder.fit(X_train[["CLIP"]])
X_train["CLIP"] = encoder.transform(X_train[["CLIP"]])

print(X_train["Okuda"].value_counts())
encoder = OrdinalEncoder(categories=[["Stage I","Stage II","Stage III"]])
encoder.fit(X_train[["Okuda"]])
X_train["Okuda"] = encoder.transform(X_train[["Okuda"]])

print(X_train["TNM"].value_counts())
encoder = OrdinalEncoder(categories=[["Stage-I","Stage-II","Stage-IIIA","Stage-IIIB","Stage-IIIC", "Stage-IVA", "Stage-IVB"]])
encoder.fit(X_train[["TNM"]])
X_train["TNM"] = encoder.transform(X_train[["TNM"]])

print(X_train["BCLC"].value_counts())
encoder = OrdinalEncoder(categories=[["Stage-A","Stage-B","Stage-C","Stage-D"]])
encoder.fit(X_train[["BCLC"]])
X_train["BCLC"] = encoder.transform(X_train[["BCLC"]])

# Añadim les variables qualitatives tractades amb get_dummies i llevam les anteriors
X_train = pandas.concat([X_train, dummies_hepatitis,dummies_pathology,dummies_tumor_nodul], axis = 1)

df = X_train.drop(columns=['tumor_nodul','hepatitis','Pathology'])

X_train.info()
#=======================================================================================================================

#Elecció alpha
# alphas = 10**np.linspace(10,-2,100)*0.5
# lasso = GroupLasso(n_iter= 10000)
# coefs = []

# for a in alphas:
#     lasso.set_params(alpha=a)
#     lasso.fit(X_train, y_train)
#     coefs.append(lasso.coef_)
    
# ax = plt.gca()
# ax.plot(alphas*2, coefs)
# ax.set_xscale('log')
# plt.axis('tight')
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.show()
