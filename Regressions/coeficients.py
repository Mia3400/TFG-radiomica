from statistics import mean
import numpy as np



#Aquestes funcions ens ajudarán a calcular el coeficient de determinació r2 i l'AIC dels nostres models
def SSR(y,prediccions):
    lista = []
    for i in range(0,len(y)):
        lista.append((y[i]-prediccions[i])**2)
    return sum(lista)
def SST(y):
    y_mean = mean(y)
    lista =[]
    for y in y:
        lista.append((y-y_mean)**2)
    return sum(lista)
def r2(model, X,y):
    prediccions = model.predict(X)
    y = list(y.loc[:,"OS"])
    return 1-(SSR(y,prediccions)/SST(y))


def calculate_aic(modelo,X,y):
    n = len(y)
    k = len(modelo.coef_)
    y_pred = modelo.predict(X)
    residus = y.loc[:,"OS"].tolist() - y_pred
    ssr = np.sum(residus**2)
    ver = -(n/2) * (np.log(2*np.pi) + np.log(ssr/n) + 1)
    aic = 2 * k - 2 * ver
    return aic

