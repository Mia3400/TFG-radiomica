from sklearn import linear_model
from Extraccio_dades.feature_data import feature_data,get_seg_ct_filepath,feature_extractor
from predict_feature import predict_feature
import pandas
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#Aqui utilitzam totes les funcions creades fins ara per fer la regressió

clinical_data ="C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx"
filepath = "C:/TFG-code/manifest-1680446438379/"

#Si tenim un metadata nou haurem de descomentar aquestes linees i comentar la següent:
#data = feature_data(filepath)
#data.to_csv("feature_data.csv")

data = pandas.read_csv("C:/Users/miacr/TFG-radiomica/feature_data.csv").set_index("Patient")

names = data.index.to_list()
predict = predict_feature(clinical_data, names)


X_train, X_test, y_train, y_test = train_test_split(data, predict, test_size=0.1, random_state=0)
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(y_pred)
print(y_test)
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))