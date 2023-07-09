import pandas 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
from coeficients import r2,calculate_aic
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
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
#Extracció de dades:
#=========================================
clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
filepath = "C:/TFG-code/manifest-1643035385102/"


radiomic_data = pandas.read_csv("C:/Users/miacr/TFG-radiomica/feature_data_2.csv")
clinical_data = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_processed.csv")
clinical_data = clinical_data.rename(columns={"TCIA_ID":"Patient"})
data = pandas.merge(radiomic_data,clinical_data, on="Patient").set_index("Patient")

names = data.index.to_list()                        #Llista amb l'ID del pacients que tenen informacio necessaria
predict = predict_feature(clinical_data_path, names)     #OS dels mateixos pacients

#Anàlisi exploratori de dades:
#detectam variables qualificatives, serán just les mateixes que al clinical data
for col in data.columns:
    unique_vals = data[col].unique()
    if len(unique_vals) <= 5:
        data.loc[:,col] = data.loc[:,col].astype("category")
print(data.head)
data.info()

#Correlació:

corr_matrix = data.corr()
high_corr = corr_matrix[corr_matrix.abs() > 0.8]
correlated_vars = [(col1, col2) for col1 in high_corr.columns for col2 in high_corr.index if col1 != col2 and  high_corr.loc[col2, col1] > 0.8]
var_counts = {}
for col1, col2 in correlated_vars:
    var_counts[col1] = var_counts.get(col1, 0) + 1
    var_counts[col2] = var_counts.get(col2, 0) + 1
print(var_counts)
for var1,var2 in correlated_vars:
    if var1 in data.columns and var2 in data.columns:
        if var_counts.get(var1)<= var_counts.get(var2):
            data = data.drop(var1,axis=1)
        else:
            data = data.drop(var2,axis=1)

#Separació:
X_train, X_test, y_train, y_test = train_test_split(data, predict, train_size   = 0.7,shuffle = True, random_state= 1234)

#Valors NA's
X_train = X_train.drop(X_train.columns[X_train.isna().any()],1)
X_train.info()
#Estandarització:

for colname in X_test.columns:
    if colname not in X_train.columns:
        X_test = X_test.drop(colname, axis = 1)

X_train_num = X_train.select_dtypes(include=['float64', 'int'])
col_num = X_train_num.columns


scaler = StandardScaler()
X_train.loc[:,col_num] = scaler.fit_transform(X_train.loc[:,col_num])
X_test.loc[:,col_num] = scaler.fit_transform(X_test.loc[:,col_num])

#Elecció alpha
alphas = np.linspace(0.01,34,50)
lasso = Lasso(max_iter=15000)
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
plt.title('coefficients Lasso en functió d\'alpha')
plt.show()

zip_alphas = (list(zip(coefs, alphas)))
nonzero_coefs = []
for coef, alpha in zip_alphas:
    n_nonzero = np.sum(coef != 0)
    nonzero_coefs.append((alpha, n_nonzero))

# cross-validation per l'elecció:
alphas_propostes = np.linspace(0.1,1,20)
Lassoreg= LassoCV(alphas= alphas_propostes,cv = 4,random_state=1234).fit(X_train,y_train)
print("L'hiperparàmetre elegit és" , Lassoreg.alpha_)

Lasso_reg= Lasso(alpha=1, random_state=1234)
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

#Avaluació del model
#=======================================================
# metriques amb training
scoring = "r2"
results = cross_val_score(Lasso_reg, X_train, y_train, cv=4, scoring=scoring)
print("R squared al subconjunt de training val: ", results.mean())

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

axes[1, 0].set_title('Distribució residus del model', fontsize = 10,
                     fontweight = "bold")

axes[1, 0].set_xlabel("Residu")
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

#Dades test
prediccions = Lasso_reg.predict(X_test)
df_predicciones = pandas.DataFrame({'OS' : y_test.loc[:,"OS"], 'predicció' : prediccions})
print(df_predicciones)

r2_computada = r2(Lasso_reg, X_test,y_test)
print(r2_computada)
print(r2_score(y_pred= prediccions, y_true=y_test))
print("AIC")
print(calculate_aic(modelo=Lasso_reg,X = X_test, y= y_test))

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