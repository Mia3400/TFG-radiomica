import pandas

df = pandas.read_excel("C:/Users/miacr/OneDrive/Documentos/TFG/HCC-TACE-Seg_clinical_data-V2.xlsx")
predict_feature = df.loc[:,["TCIA_ID","OS"]].set_index('TCIA_ID')
print(predict_feature)

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