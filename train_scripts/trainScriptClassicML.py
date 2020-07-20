import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


# Read data 
df = pd.read_csv("./data/train.csv/train.csv")



print('-'*15, "data loaded!")

# Replace inf values with np.nan, then replace nan with 0
df.replace([np.inf, -np.inf], np.nan,inplace=True)
df = df.fillna(0)

# Features
X = df.drop(['sample_id', 'y'], axis=1)
# Labels
y = df['y']


# Features normalization
features_norm = StandardScaler() 
X_std = features_norm.fit_transform(X) 

# Split data in train/test
X_train, x_test, Y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)   


models = [SVC(probability = True), XGBClassifier(), RandomForestClassifier()]

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

models_ens = list(zip(['SVM', 'XGB', 'RF'], models))
model_ens = VotingClassifier(estimators = models_ens, voting = 'soft')
model_ens.fit(X_train, Y_train)
pred = model_ens.predict(x_test)
prob_voting = model_ens.predict_proba(x_test)[:,1]

roc_auc_soft = roc_auc_score(y_test, prob_voting)

print('-'*15, "AUC score classic ML = ", roc_auc_soft)

# Plotting ROC-AUC
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, prob_voting)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('./aucClassicML.jpg')