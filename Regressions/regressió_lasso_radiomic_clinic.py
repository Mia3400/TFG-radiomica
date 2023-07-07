import pandas 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
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
#Extracció de dades:
#======================================================
clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
filepath = "C:/TFG-code/manifest-1643035385102/"

radiomic_data = pandas.read_csv("C:/Users/miacr/TFG-radiomica/feature_data_2.csv")
clinical_data_X_train = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Xtrain_processed.csv")
clinical_data_X_test = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Xtest_processed.csv")
clinical_data_y_train = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Ytrain_processed.csv")
clinical_data_y_test = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_Ytest_processed.csv")

clinical_data_X_train= clinical_data_X_train.rename(columns={"TCIA_ID":"Patient"})
clinical_data_X_test= clinical_data_X_test.rename(columns={"TCIA_ID":"Patient"})
X_test = pandas.merge(radiomic_data,clinical_data_X_test, on="Patient").set_index("Patient")
X_train =  pandas.merge(radiomic_data,clinical_data_X_train, on="Patient").set_index("Patient")

names_Xtrain= X_train.index.to_list()
names_Xtest= X_test.index.to_list()                           #Llista amb l'ID del pacients que tenen informacio necessaria
y_train = predict_feature(clinical_data_path, names_Xtrain)     #OS dels mateixos pacients
y_test = predict_feature(clinical_data_path, names_Xtest)

#Anàlisi exploratori de dades:
#======================================================
# Modificació typer de les variables qualitatives
for col in X_train.columns:
    unique_vals = X_train[col].unique()
    if len(unique_vals) <= 7:
        X_train.loc[:,col] = X_train.loc[:,col].astype("category")
print(X_train.head)
X_train.info()

#Valors NA's
X_train = X_train.drop(X_train.columns[X_train.isna().any()],1)

#Estandarització:
#======================================================
X_train_num = X_train.select_dtypes(include=['float64', 'int'])
col_num = X_train_num.columns


scaler = StandardScaler()
X_train.loc[:,col_num] = scaler.fit_transform(X_train.loc[:,col_num])
X_test.loc[:,col_num] = scaler.fit_transform(X_test.loc[:,col_num])

## Elecció alpha
#======================================================
alphas = np.linspace(0.01,40,200)
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
plt.title('Coeficients en funció d\' alpha')
plt.show()
plt.close()

zip_alphas = (list(zip(coefs, alphas)))
nonzero_coefs = []
for coef, alpha in zip_alphas:
    n_nonzero = np.sum(coef != 0)
    nonzero_coefs.append((alpha, n_nonzero))

alphas, n_nonzero = zip(*nonzero_coefs)
plt.plot(alphas, n_nonzero)
plt.xlabel('alpha')
plt.ylabel('number of non-zero coefficients')
plt.xscale('log')
plt.show()

# Validació creuada per la elecció del paràmetre alpha
alphas_propostes = np.linspace(0.01,1,20)
Lassoreg= LassoCV(alphas= alphas_propostes,cv = 4,random_state=1234).fit(X_train,y_train)
print("L'hiperparàmetre elegit és" , Lassoreg.alpha_)

Lasso_reg= Lasso(alpha=1, random_state=1234)
Lasso_reg.fit(X_train,y_train)

results = cross_val_score(Lasso_reg, X_train, y_train, cv=4, scoring=r2)
print(results)
print("R squared training val: ", results.mean())

def print_table(data):
    print("{:<50s}{:<10s}".format("Name", "Coefficient"))
    for coef, name in data:
        print("{:<50s}{:<10f}".format(name, coef))

alpha_coefs = list(zip(Lasso_reg.coef_, X_train))

counts = {'radiomiques': 0, 'cliniques': 0}

for name, coef in alpha_coefs:
    if 'original' in coef and name != 0:
        counts['radiomiques'] += 1
    elif 'original' not in coef and name != 0:
        counts['cliniques'] += 1

print(counts)

#Avaluació del model
#======================================================
# metriques amb training
scoring = "r2"
results = cross_val_score(Lasso_reg, X_train, y_train, cv=4, scoring=scoring)
print("R squared val: ", results.mean())

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

# Avaluació a les dades test:
prediccions = Lasso_reg.predict(X_test)
df_predicciones = pandas.DataFrame({'OS' : y_test.loc[:,"OS"], 'predicció' : prediccions})
print(df_predicciones)
print(r2_score(y_true= y_test,y_pred= prediccions))
print("AIC")
print(calculate_aic(Lasso_reg,X_test,y_test))

# Gràfic d'ajust dades test:
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