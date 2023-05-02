# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the reqired packages.
2. print arrays of x and y.
3. use sigmoid function to get the output b/w 1 and 0
4.Using decision boundary plot the graph and find probabiluty and the mean value
5.Display all the outputs.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: J.Archana priya
RegisterNumber:  212221230007
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt('ex2data1.txt',delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

#Visualizing the data

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

#Sigmoid Function

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient(theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)

print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train=np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:
### Array value of x
![image](https://user-images.githubusercontent.com/93427594/235622114-e66c6876-110a-43ae-acb6-e82d5bebadb5.png)
### Array value of y
![image](https://user-images.githubusercontent.com/93427594/235622204-58ec0079-2551-4047-bb8a-7f643f33d952.png)
### Exam 1 - score graph
![image](https://user-images.githubusercontent.com/93427594/235622282-5dbea2f6-138e-48fd-9d8e-0ba3fcf96def.png)
### sigmoid function graph
![image](https://user-images.githubusercontent.com/93427594/235622428-6d0907e3-6003-40a7-a0ad-ed4dd6f786ba.png)
### x_trained_geadvalue
![image](https://user-images.githubusercontent.com/93427594/235622758-2c3f4da2-f170-4ab7-9370-4b7d8eb505f7.png)
### y_trained_geadvalue
![image](https://user-images.githubusercontent.com/93427594/235622853-f537fda4-03d2-4406-a5bf-55df80197f88.png)
### print res.x
![image](https://user-images.githubusercontent.com/93427594/235623057-4f3ef1f7-4ad3-4192-86e9-509eb6fe27fd.png)

### Decosion boundary - graph for exam score
![image](https://user-images.githubusercontent.com/93427594/235623122-ccd2b541-01f4-4610-9fed-3679f9d67c1b.png)

### probabilty value
![image](https://user-images.githubusercontent.com/93427594/235623167-39973fc7-56f8-4501-8cd3-dc1a4aaf0128.png)

### prediction value for mean
![image](https://user-images.githubusercontent.com/93427594/235623212-ccdd08d0-1fb5-4df7-8309-58eafeed34e6.png)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

