import pandas 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
from coeficients import r2
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
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

clinical_data_path ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
filepath = "C:/TFG-code/manifest-1643035385102/"

#data = feature_data(filepath)           #Empram el programa per a l'extracció de dades radiòmiques
#data.to_csv("feature_data2.csv")         #Guardam les dades a un CSV i el llegim 

radiomic_data = pandas.read_csv("C:/Users/miacr/TFG-radiomica/feature_data_2.csv")
clinical_data = pandas.read_csv("C:/Users/miacr/TFG-radiomica/clinical_data_processed.csv")
clinical_data = clinical_data.rename(columns={"TCIA_ID":"Patient"})
data = pandas.merge(radiomic_data,clinical_data, on="Patient").set_index("Patient")
#detectam variables qualificatives, serán just les mateixes que al clinical data
for col in data.columns:
    unique_vals = data[col].unique()
    if len(unique_vals) <= 5:
        data.loc[:,col] = data.loc[:,col].astype("category")
print(data.head)
data.info()
names = data.index.to_list()                        #Llista amb l'ID del pacients que tenen informacio necessaria
predict = predict_feature(clinical_data_path, names)     #OS dels mateixos pacients

#Anàlisi exploratori de dades:


#Correlació:
corr_matrix = data.corr()
high_corr = corr_matrix[corr_matrix.abs() > 0.8]
# Get the variable pairs with high correlation
correlated_vars = [(col1, col2) for col1 in high_corr.columns for col2 in high_corr.index if col1 != col2 and  high_corr.loc[col2, col1] > 0.8]
# Print the variable pairs with high correlation
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
data.info()
#Separació:
#A la pràctica sempre es sol fer 0.8 i 0.2, però tenc molt poques mostres, no se si es petit encare que després fassem cross validation
X_train, X_test, y_train, y_test = train_test_split(data, predict, train_size   = 0.7,shuffle = True, random_state= 1234)
#Valors NA's
#print(X_train.columns[X_train.isna().any()], X_train.columns[X_train.isna().any()].isna().sum())
X_train = X_train.drop(X_train.columns[X_train.isna().any()],1)

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
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of alpha')
plt.show()

zip_alphas = (list(zip(coefs, alphas)))
nonzero_coefs = []
for coef, alpha in zip_alphas:
    n_nonzero = np.sum(coef != 0)
    nonzero_coefs.append((alpha, n_nonzero))

# Plot the number of non-zero coefficients as a function of alpha
alphas, n_nonzero = zip(*nonzero_coefs)
plt.plot(alphas, n_nonzero)
plt.xlabel('alpha')
plt.ylabel('number of non-zero coefficients')
plt.xscale('log')
plt.show()

Lasso_reg= Lasso(alpha=0.9, random_state=42)
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
# metriques amb training
scoring = "neg_mean_squared_error"
results = cross_val_score(Lasso_reg, X_train, y_train, cv=4, scoring=scoring)
print("Mean Squared Error: ", results.mean())

scoring = "r2"
results = cross_val_score(Lasso_reg, X_train, y_train, cv=4, scoring=scoring)
print("R squared val: ", results.mean())

print(r2_score(
    y_pred= Lasso_reg.predict(X_train),
    y_true=  y_train))
print("R2 sense cross-val computat per jo:")
# print(r2(Lasso_reg, X_train,y_train))
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
axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
axes[0, 0].set_xlabel('Real')
axes[0, 0].set_ylabel('Predicción')
axes[0, 0].tick_params(labelsize = 7)

axes[0, 1].scatter(list(range(len(y_train))), y_train.loc[:,"OS"].tolist() - cv_prediccones,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
axes[0, 1].set_xlabel('id')
axes[0, 1].set_ylabel('Residuo')
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

axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10,
                     fontweight = "bold")
axes[1, 0].set_xlabel("Residuo")
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
axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
axes[1, 1].tick_params(labelsize = 7)

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold");
plt.show()

#Datos test
prediccions = Lasso_reg.predict(X_test)
df_predicciones = pandas.DataFrame({'OS' : y_test.loc[:,"OS"], 'predicció' : prediccions})
print(df_predicciones)
mse = mean_squared_error(y_true = y_test,y_pred = prediccions)
print(mse)
r2_computada = r2(Lasso_reg, X_test,y_test)
print(r2_computada)
print(r2_score(y_pred= prediccions, y_true=y_test))