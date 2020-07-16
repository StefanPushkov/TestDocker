import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils import data as data_utils
from sklearn.preprocessing import StandardScaler 
import torch.nn.functional as F

print(15*'-', " moduleNN.py logs ", 15*'-')
print("Train neural network")

# Load data
data = pd.read_csv('../data/train.csv/train.csv')

# Replace inf values with nan, then replace nan with 0
data.replace([np.inf, -np.inf], np.nan,inplace=True)
data = data.fillna(0)

# Get features and label separately 
X = data.drop(['sample_id', 'y'], axis=1)
y = data['y']

# Split in train/test
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Features normalization
scale_features_std = StandardScaler() 
X_train_std = scale_features_std.fit_transform(X_train) 
x_test_std = scale_features_std.fit_transform(x_test)

# Convert np.array to torch tensor: train and test set
X_train_tensor = torch.tensor(X_train_std, dtype=torch.float)
Y_train_tensor = torch.tensor(Y_train.values).flatten() 

x_test_tensor = torch.tensor(x_test_std, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values).flatten() 

# Create train dataloader
batch_size = 128

train_dataset = data_utils.TensorDataset(X_train_tensor, Y_train_tensor) 
train_loader = data_utils.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)


# Define model
class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet,self).__init__()
        
        self.fc1 = nn.Linear(1612,3200) # input size: 1612 features
        self.fc2 = nn.Linear(3200,3200)
        self.fc3 = nn.Linear(3200,1600)
        self.fc4 = nn.Linear(1600,1) # output size: 1 (1 or 0)
        
        self.bn0 = nn.BatchNorm1d(1612)
        self.bn1 = nn.BatchNorm1d(3200)
        self.bn2 = nn.BatchNorm1d(3200)
        self.bn3 = nn.BatchNorm1d(1600)
        
        
    # Forward pass
    def forward(self,x):
        x = self.bn0(x)
        x = F.relu(self.fc1(x))

        x = self.bn1(x)
        x = F.relu(self.fc2(x))

        x = self.bn2(x)
        x = F.relu(self.fc3(x))

        x = self.bn3(x)        
        x = self.fc4(x)
        return x
        
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x))
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)