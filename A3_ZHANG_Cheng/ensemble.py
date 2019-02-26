# ensembled model by majority vote of 10 model results
import pandas as pd

models_loc = ["experiment/kaggle"+str(i)+".csv" for i in range(1, 11)]

models_combined=pd.DataFrame()
for i in range(1, 11):
    loc = models_loc[i-1]
    f = pd.read_csv(loc)
    models_combined["Id"] = f["Id"].values
    models_combined["Category"+str(i)] = f["Category"].values

cols = list(models_combined.columns)
cols = cols[1:]

models_combined["Category"] = models_combined[cols].mode(axis=1)[0].astype(int)
models_combined[["Id", "Category"]].to_csv("experiment/ensemble2.csv", index=False)
