#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np


# In[48]:


data=pd.read_csv(r"AQI_2022_06.csv")


# In[49]:


data.nunique()


# In[50]:


data.isnull().sum()


# In[51]:


data.columns


# In[ ]:


data=data.drop(['"unit"','"pollutant"','"datacreationdate"','"winddirec"','"windspeed"'],axis=1)


# In[53]:


data=data.dropna()


# In[54]:


data


# In[56]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data['"sitename"'] = label_encoder.fit_transform(data['"sitename"'])
data['"county"'] = label_encoder.fit_transform(data['"county"'])
data['"status"'] = label_encoder.fit_transform(data['"status"'])


# In[57]:


data


# In[58]:


print(data.dtypes)


# In[59]:


from sklearn.preprocessing import StandardScaler

y = data['"aqi"'].values
# Create a StandardScaler object
scaler = StandardScaler()
# Use the fit_transform method to standardize the data
y = scaler.fit_transform(y.reshape(-1, 1))
X=data.drop(['"aqi"'],axis=1)


# In[60]:


y


# In[79]:


data.columns


# In[82]:


from sklearn.ensemble import RandomForestRegressor


rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y)
feature_importances = rf_regressor.feature_importances_


feature_names = X.columns 

sorted_indices = np.argsort(feature_importances)[::-1]


for idx in sorted_indices:
    print(f"{feature_names[idx]}: {feature_importances[idx]:.4f}")


plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices], align='center')
plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis() 
plt.show()



# In[61]:


# Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)



# In[62]:


y_test


# In[63]:


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Divide the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a Random Forest Regressor
rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)
rf_pred = rf_regressor.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mape = np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
print("RF Regressor - MSE: {:.2f}, R^2: {:.2f}, MAE: {:.2f}".format(rf_mse, rf_r2, rf_mae))

from sklearn.linear_model import LinearRegression

# Linear Regression model
lr_regressor = LinearRegression()
lr_regressor.fit(X_train, y_train)
lr_pred = lr_regressor.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mape = np.mean(np.abs((y_test - lr_pred) / y_test)) * 100
print("Linear Regressor - MSE: {:.2f}, R^2: {:.2f}, MAE: {:.2f}".format(lr_mse, lr_r2, lr_mae))


# In[65]:


# Divide the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)


# In[77]:


import torch
import torch.nn as nn
import torch.optim as optim

class BP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Define network parameters
input_size = X_train.shape[1]
hidden_size = 100  
output_size = 1
# Create a network instance
model = BP(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500

# Initialize an empty list to hold the loss values
losses = []

for epoch in range(num_epochs):
    # Convert data to torch tensor
    inputs = torch.tensor(X_train, dtype=torch.float32).to(device)
    targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    #Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Loss Curve
plt.plot(range(1, num_epochs+1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()


# In[67]:


inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
targets = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
bp_pred = model(inputs).detach().cpu().numpy()
bp_mse = mean_squared_error(y_test, bp_pred)
bp_r2 = r2_score(y_test, bp_pred)
bp_mae = mean_absolute_error(y_test, bp_pred)
bp_mape = np.mean(np.abs((y_test - bp_pred) / y_test)) * 100

print("BP Regressor - MSE: {:.2f}, R^2: {:.2f}, MAE: {:.2f}".format(bp_mse, bp_r2, bp_mae))


# In[ ]:


bp_pred = scaler.inverse_transform(bp_pred.reshape(-1,1))
rf_pred=scaler.inverse_transform(rf_pred.reshape(-1,1))
lr_pred=scaler.inverse_transform(lr_pred.reshape(-1,1))
y_test=scaler.inverse_transform(y_test.reshape(-1,1))


# In[70]:


start = 10  
end =150   

y_test_subset = y_test[start:end]
bp_pred_subset = bp_pred[start:end]
rf_pred_subset = rf_pred[start:end]
lr_pred_subset = lr_pred[start:end]


plt.figure(figsize=(10, 6))
plt.plot(y_test_subset, label='True Values', linewidth=2)
plt.plot(bp_pred_subset, label='BP Predictions', linestyle='--')
plt.plot(rf_pred_subset, label='RF Predictions', linestyle='-.')
plt.plot(lr_pred_subset, label='LR Predictions', linestyle=':')
plt.title('Model Predictions vs True Values')
plt.legend()
plt.show()


# In[75]:


plt.figure(figsize=(18, 6))

# BP
plt.subplot(1, 3, 1)
plt.scatter(bp_pred_subset, y_test_subset, alpha=0.5)
plt.xlabel('BP Predictions')
plt.ylabel('True Values')
plt.title('BP Neural Network')

# forest
plt.subplot(1, 3, 2)
plt.scatter(rf_pred_subset, y_test_subset, alpha=0.5)
plt.xlabel('RF Predictions')
plt.title('Random Forest')

# linear
plt.subplot(1, 3, 3)
plt.scatter(lr_pred_subset, y_test_subset, alpha=0.5)
plt.xlabel('LR Predictions')
plt.title('Linear Regression')

plt.tight_layout()
plt.show()


# In[76]:


import matplotlib.pyplot as plt

min_val = min(y_test_subset)
max_val = max(y_test_subset)

# Draw a scatter plot of the predicted value and the true value of the BP neural network model, and add a diagonal lineplt.figure(figsize=(10, 6))
plt.scatter(y_test_subset, bp_pred_subset, label='BP Predictions', marker='o')
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('BP Neural Network Predictions vs True Values')
plt.legend()
plt.show()

# Draw a scatter plot of the predicted values ​​and true values ​​of the random forest model and add a diagonal lineplt.figure(figsize=(10, 6))
plt.scatter(y_test_subset, rf_pred_subset, label='RF Predictions', marker='x')
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Predictions vs True Values')
plt.legend()
plt.show()

# Draw a scatter plot of the predicted value and the true value of the linear regression model, and add a diagonal line
plt.figure(figsize=(10, 6))
plt.scatter(y_test_subset, lr_pred_subset, label='LR Predictions', marker='^')
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Predictions vs True Values')
plt.legend()
plt.show()


# In[73]:


import matplotlib.pyplot as plt

# Draw a scatter plot of the predicted value and the true value of the BP neural network model
plt.figure(figsize=(7, 4))
plt.scatter(y_test_subset, bp_pred_subset, label='BP Predictions', marker='o')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('BP Neural Network Predictions vs True Values')
plt.legend()
plt.show()

# Draw a scatter plot of the predicted values ​​and true values ​​of the random forest model
plt.figure(figsize=(7, 4))
plt.scatter(y_test_subset, rf_pred_subset, label='RF Predictions', marker='x')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Predictions vs True Values')
plt.legend()
plt.show()

# Draw a scatter plot of the predicted values ​​and true values ​​of the linear regression model
plt.figure(figsize=(7, 4))
plt.scatter(y_test_subset, lr_pred_subset, label='LR Predictions', marker='^')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Predictions vs True Values')
plt.legend()
plt.show()

