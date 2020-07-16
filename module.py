import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump



data = pd.read_csv("./data/train.csv/train.csv")

print("data loaded!")

data.replace([np.inf, -np.inf], np.nan,inplace=True)

data = data.fillna(0)

X = data.drop(['sample_id', 'y'], axis=1)
y = data['y']


# Data normalization
from sklearn.preprocessing import StandardScaler 
scale_features_std = StandardScaler() 
X_std = scale_features_std.fit_transform(X)     

X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(max_depth=5, random_state=42)
clf.fit(X_train, Y_train)

y_preds = clf.predict(x_test)


AUC = metrics.roc_auc_score(y_test, y_preds)

print("RESULT: auc = {0}".format(AUC))

dump(clf, './models/rf.joblib')
print("Random Forest classifier saved!")