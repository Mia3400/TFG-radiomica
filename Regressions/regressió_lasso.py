import pandas 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from Extraccio_dades.feature_data import feature_data
from Extraccio_dades.predict_feature import predict_feature
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Extraccio_dades.predict_feature import predict_feature
from sklearn.preprocessing import scale 


#Extracció de dades:

clinical_data ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
filepath = "C:/TFG-code/manifest-1643035385102/"

data = feature_data(filepath)           #Empram el programa per a l'extracció de dades radiòmiques
data.to_csv("feature_data2.csv")         #Guardam les dades a un CSV i el llegim 

data = pandas.read_csv("C:/Users/miacr/TFG-radiomica/feature_data2.csv").set_index("Patient")

names = data.index.to_list()                        #Llista amb l'ID del pacients que tenen informacio necessaria
predict = predict_feature(clinical_data, names)     #OS dels mateixos pacients

#Mini-anàlisi exploratori de dades:
print(data.shape)
print(data.isna().sum().sort_values())

X_train, X_test, y_train, y_test = train_test_split(data, predict, test_size=0.2, random_state=0)




#Normalitzar dades per LASSO regression
# ss = StandardScaler()
# ndata = ss.fit_transform(data)

# clinical_data ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
# names = data.index.to_list()
# predict = predict_feature(clinical_data, names)

# X_train, X_test, y_train, y_test = train_test_split(ndata, predict, test_size=0.2, random_state=0)

# alphas = 10**np.linspace(10,-2,100)*0.5
# lasso = Lasso(max_iter = 10000)
# coefs = []

# for a in alphas:
#     lasso.set_params(alpha=a)
#     lasso.fit(scale(X_train), y_train)
#     coefs.append(lasso.coef_)
    
# ax = plt.gca()
# ax.plot(alphas*2, coefs)
# ax.set_xscale('log')
# plt.axis('tight')
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.show()

# lasso_fit =lasso.set_params(alpha=1.04)
# lasso_fit.fit(X_train, y_train)
# mse = mean_squared_error(y_test, lasso.predict(X_test))

# print( lasso_fit.coef_, lasso_fit.score, np.sqrt(mse))
# print(pandas.Series(lasso.coef_, index= data.columns))

# lasso = Lasso(max_iter=10000)
# lassocv = LassoCV( cv=10, max_iter=100000)
# lassocv.fit(X_train, y_train)
# lasso.set_params(alpha=lassocv.alpha_)
# print("Alpha=", lassocv.alpha_)
# lasso.fit(X_train, y_train)
# print("mse = ",mean_squared_error(y_test, lasso.predict(X_test)))
# print("best model coefficients:")
# print(pandas.Series(lasso.coef_, index= data.columns))

# X_train, X_test, y_train, y_test = train_test_split(data, predict, test_size=0.15, random_state=0)
# alphas = np.linspace(0.01,500,100)
# lassocv = LassoCV(alphas=alphas, cv=5).fit(X_train,y_train)
# print(lassocv)
# score = lassocv.score(X_train,y_train)
# ypred = lassocv.predict(X_test)
# mse = mean_squared_error(y_test,ypred)
# print("Alpha:{0:.2f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}"
#     .format(lassocv.alpha_, score, mse, np.sqrt(mse)))

# x_ax = range(len(X_test))
# plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
# plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
# plt.legend()
# plt.show() 
