### EXP NO: 01

### .

# <p align = "center"> Developing a Neural Network Regression Model </p>
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The Neural network model contains input layer,two hidden layers and output layer.Input layer contains a single neuron.Output layer also contains single neuron.First hidden layer contains six neurons and second hidden layer contains seven neurons.A neuron in input layer is connected with every neurons in a first hidden layer.Similarly,each neurons in first hidden layer is connected with all neurons in second hidden layer.All neurons in second hidden layer is connected with output layered neuron.Relu activation function is used here .It is linear neural network model(single input neuron forms single output neuron).

## Neural Network Model
![Screenshot (389)](https://user-images.githubusercontent.com/75243072/187078981-2aafe51a-eaff-4dd6-a902-e6f6bc567333.png)

## DESIGN STEPS
### STEP 1:
Loading the dataset
### STEP 2:
Split the dataset into training and testing
### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.
### STEP 4:
Build the Neural Network Model and compile the model.
### STEP 5:
Train the model with the training data.
### STEP 6:
Plot the performance plot
### STEP 7:
Evaluate the model with the testing data.

## PROGRAM
```python
# Developed By:KUMARAN.B
# Register Number:212220230026

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv("dl2.csv")
df.head()
x=df[['input']].values
x
y=df[['output']].values
y
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=40)

scaler=MinMaxScaler()
scaler.fit(xtrain)
scaler.fit(xtest)
xtrain1=scaler.transform(xtrain)
xtest1=scaler.transform(xtest)

model=Sequential([
    Dense(6,activation='relu'),
    Dense(7,activation='relu'),
    Dense(1)
])
model.compile(optimizer='rmsprop',loss='mse')
model.fit(xtrain1,ytrain,epochs=4000)
lossmodel=pd.DataFrame(model.history.history)
lossmodel.plot()
model.evaluate(xtest1,ytest)

xn1=[[77]]
xn11=scaler.transform(xn1)
model.predict(xn11)
```

## Dataset Information
![Screenshot (385)](https://user-images.githubusercontent.com/75243072/187077397-7c129470-1f53-475f-ac8e-a755c425bb9b.png)

## OUTPUT
### Training Loss Vs Iteration Plot
![Screenshot (386)](https://user-images.githubusercontent.com/75243072/187077611-8f14ddac-da56-4067-9f51-564cd37bf1a1.png)

### Test Data Root Mean Squared Error
![Screenshot (387)](https://user-images.githubusercontent.com/75243072/187080899-276e0eed-c3c9-4d40-9c9b-d7d7935acb5a.png)

### New Sample Data Prediction
![Screenshot (388)](https://user-images.githubusercontent.com/75243072/187077541-a8b68c4c-f3e7-4780-a890-758f0da449db.png)

## RESULT
Thus,the neural network regression model for the given dataset is developed.
