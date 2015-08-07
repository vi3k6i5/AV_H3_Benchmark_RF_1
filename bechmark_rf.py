import pandas as pd
import numpy as np

def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

X_train = pd.read_csv("Train.csv")
X_test = pd.read_csv("Test.csv")
X_train = shuffle(X_train)

X_train["is_train"] = 1
X_test["is_train"] = 0
panel = pd.concat([X_train, X_test])

#Data cleaning
panel.drop(["institute_latitude"], axis=1, inplace=True)
panel.drop(["institute_longitude"], axis=1, inplace=True)
panel.drop(["institute_city"], axis=1, inplace=True)
panel.drop(["institute_zip"], axis=1, inplace=True)
panel.drop(["institute_country"], axis=1, inplace=True)

panel.drop(["Unnamed: 26"], axis=1, inplace=True)

panel.drop(["secondary_subject"], axis=1, inplace=True)
panel.drop(["secondary_area"], axis=1, inplace=True)

#convert categorical features to values
Var4_normalized = pd.Categorical.from_array(panel['Var4'])
institute_state_normalized = pd.Categorical.from_array(panel['institute_state'])
Var8_normalized = pd.Categorical.from_array(panel['Var8'])
Var15_normalized = pd.Categorical.from_array(panel['Var15'])
project_subject_normalized = pd.Categorical.from_array(panel['project_subject'])
subject_area_normalized = pd.Categorical.from_array(panel['subject_area'])
Resource_Category_normalized = pd.Categorical.from_array(panel['Resource_Category'])
Resource_Sub_Category_normalized = pd.Categorical.from_array(panel['Resource_Sub_Category'])
Var23_normalized = pd.Categorical.from_array(panel['Var23'])
Var24_normalized = pd.Categorical.from_array(panel['Var24'])

panel['Var4_normalized'] = Var4_normalized.codes
panel['institute_state_normalized'] = institute_state_normalized.codes
panel['Var8_normalized'] = Var8_normalized.codes
panel['Var15_normalized'] = Var15_normalized.codes
panel['project_subject_normalized'] = project_subject_normalized.codes
panel['subject_area_normalized'] = subject_area_normalized.codes
panel['Resource_Category_normalized'] = Resource_Category_normalized.codes
panel['Resource_Sub_Category_normalized'] = Resource_Sub_Category_normalized.codes
panel['Var23_normalized'] = Var23_normalized.codes
panel['Var24_normalized'] = Var24_normalized.codes

panel.drop(["Var4"], axis=1, inplace=True)
panel.drop(["institute_state"], axis=1, inplace=True)
panel.drop(["Var8"], axis=1, inplace=True)
panel.drop(["Var15"], axis=1, inplace=True)
panel.drop(["project_subject"], axis=1, inplace=True)
panel.drop(["subject_area"], axis=1, inplace=True)
panel.drop(["Resource_Category"], axis=1, inplace=True)
panel.drop(["Resource_Sub_Category"], axis=1, inplace=True)
panel.drop(["Var23"], axis=1, inplace=True)
panel.drop(["Var24"], axis=1, inplace=True)

#convert boolean features
d = {'Y': 1, 'N': 0}

panel['Var10'] = panel['Var10'].map(d)
panel['Var11'] = panel['Var11'].map(d)
panel['Var12'] = panel['Var12'].map(d)
panel['Var13'] = panel['Var13'].map(d)
panel['Var14'] = panel['Var14'].map(d)
panel['Instructor_Past_Performance'] = panel['Instructor_Past_Performance'].map(d)
panel['Instructor_Association_Industry_Expert'] = panel['Instructor_Association_Industry_Expert'].map(d)

X_train = panel[panel["is_train"]==1]
X_test = panel[panel["is_train"]==0]

X_train = X_train.copy()
X_test = X_test.copy()
y_train = X_train.pop("Project_Valuation")
y_test = X_test.pop("Project_Valuation")
id_train = X_train.pop("ID")
id_test = X_test.pop("ID")

#Trying RF on the model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=42,max_features="auto", 
                              min_samples_leaf=7)
model.fit(X_train, y_train)
output = pd.DataFrame(model.predict(X_test))

output.columns = ['Project_Valuation']
output["ID"] = id_test.values
output = output[['ID', 'Project_Valuation']] 
output.head()
output.to_csv('benchmark_1_rf.csv', sep=',', index=False)

#please note that the paramets for RF have not been optimized
#Basically its just the v1 of this code.
#public leaderboard score with this code is 541.653717087
