import pandas


def predict_feature(path_clinical_data, names):
    predict = []
    clinical_data = pandas.read_excel(path_clinical_data)
    for i in range (0,len(clinical_data)):
        if clinical_data.loc[i,"TCIA_ID"] in names:
            predict.append(clinical_data.loc[i,["TCIA_ID","OS"]])
    predict_data = pandas.DataFrame(predict).set_index("TCIA_ID")
    return predict_data


#                  OS
# TCIA_ID
# HCC_001  350.285714
# HCC_002   25.285714
# HCC_003   21.428571
# HCC_004   10.285714
# HCC_005   56.285714
# ...             ...
# HCC_101  135.857143
# HCC_102  126.142857
# HCC_103  162.857143
# HCC_104   85.714286
# HCC_105  119.285714