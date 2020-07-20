import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils import data as data_utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F


# Read data 
df = pd.read_csv("./data/train.csv/train.csv")
df_test = pd.read_csv('./data/test.csv/test.csv')


print('-'*15, "Data loaded")

# Replace inf values with np.nan, then replace nan with 0
df.replace([np.inf, -np.inf], np.nan,inplace=True)
df = df.fillna(0)

df_test.replace([np.inf, -np.inf], np.nan,inplace=True)
df_test = df_test.fillna(0) 

# Features
X = df.drop(['sample_id', 'y'], axis=1)
X_submission = df_test.drop(['sample_id'], axis=1)

# Labels
y = df['y']


# Features normalization
features_norm = StandardScaler() 
X_std = features_norm.fit_transform(X) 
X_submission_std = features_norm.fit_transform(X_submission) 


# Split data in train/test
X_train, x_test, Y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)   

# To torch tensor: Train
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
Y_train_tensor = torch.tensor(Y_train.values).flatten() 

# Test
x_test_tensor = torch.tensor(x_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values).flatten() 

# Create train dataloader
batch_size = 128

train_dataset = data_utils.TensorDataset(X_train_tensor, Y_train_tensor) 
train_loader = data_utils.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)

# Create eval dataloader

eval_dataset = data_utils.TensorDataset(x_test_tensor, y_test_tensor) 
eval_loader = data_utils.DataLoader(dataset = eval_dataset, batch_size = batch_size, shuffle = True)


import torch.nn.functional as F

# Class must extend nn.Module
class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier,self).__init__()
        # Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
         
        self.fc1 = nn.Linear(1612,200)
        self.fc2 = nn.Linear(200,100)
        self.layer_out = nn.Linear(100,1)
        
        self.dropout = nn.Dropout()
        
        
        
        self.bn0 = nn.BatchNorm1d(1612)
        self.bn1 = nn.BatchNorm1d(200)
        
        self.bn_out = nn.BatchNorm1d(100)
        
        
        
    
    def forward(self,x):
        
        # Batch normalization
        x = self.bn0(x)
        
        # This applies Linear transformation to input data with non-linear activation
        x = F.relu(self.fc1(x))
        
        # Dropout
        x = self.dropout(x) 
        
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x) 
        
        
        x = self.bn_out(x)
        #This applies linear transformation to produce output data
        x = self.layer_out(x)
        
        return x
        
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self, x):
        with torch.no_grad():
            y_pred = model(x)
            y_pred_tag = torch.round(torch.sigmoid(y_pred))
        return torch.tensor(y_pred_tag, dtype=float)
    
    
    def predict_proba(self, x):
        with torch.no_grad():
            y_pred = model(x)
            prob = torch.sigmoid(y_pred)    
        return torch.tensor(prob, dtype=float)



def train_model(model, optim, criterion, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    for x, y in train_dl:
        batch = y.shape[0]
        output = model(x)   
        loss = criterion(output, y.unsqueeze(1))   
        optim.zero_grad()
        loss.backward()
        
        # Clip gradient 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optim.step()
        
        # Accumulate epoch loss 
        total += batch
        sum_loss += batch*(loss.item())
        # print("Batch loss: ", batch*(loss.item()))
    return sum_loss/total



if __name__ == '__main__':
    # Initialize the model        
    model = MyClassifier()
    # Define loss criterion
    criterion = nn.BCEWithLogitsLoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(15*'-', 'Model started training')
    #Number of epochs
    epochs = 150
    #List to store losses
    train_losses = []
    for i in range(epochs):
        epoch_loss = train_model(model=model, optim=optimizer, criterion=criterion, train_dl=train_loader)
        train_losses.append(epoch_loss)
        if i % 10 == 0:
            print("Epoch {0}, Loss {1}".format(i+1, epoch_loss))

    auc_sc = roc_auc_score(y_test_tensor.long(), model.predict_proba(x_test_tensor))

    print('-'*15, "AUC score Network = ", auc_sc)
    
    prob_voting = model.predict_proba(x_test_tensor)

    # Plotting ROC-AUC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_tensor.numpy(), prob_voting.numpy())
    roc_auc = auc(false_positive_rate, true_positive_rate)

        
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('./aucNN.jpg')
    
    # Convert numpy to torch tensor and make prediction
    X_submission_tensor = torch.tensor(X_submission_std, dtype=torch.float)
    a = model.predict_proba(X_submission_tensor).numpy()
    
    # Create submission
    submission = pd.DataFrame(df_test["sample_id"], index=None)
    submission["y"] = a
    submission.to_csv("./Stef_sabmission.csv", sep=",", index=False)

    print("Submission created (NN).")
    
