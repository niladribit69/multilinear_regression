#1.importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#2.loading the dataset
mydata=pd.read_csv('50_Startups.csv')
X=mydata.iloc[:,:-1].values
Y=mydata.iloc[:,4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder   #onehotencoder only works on numerical data
le_X=LabelEncoder()
X[:,3]=le_X.fit_transform(X[:,3])
oneh=OneHotEncoder(categorical_features=[3])
X=oneh.fit_transform(X).toarray() #doesnt support input location if not.toarray()

#avoiding dummy variable trap
X=X[:,1:]

#splitting into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0) #to have same output in different system

#4.applying multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_train)            #predict creates new values

#OPTIMIZING MODEL USING BACKWARD ELIMINATION
X_zero=np.ones((50,1))
X=np.append(X_zero,X,axis=1)
import statsmodels.formula.api as sm

#REMOVING HIGHEST P VALUE
X_opt=X
regress=sm.OLS(endog=Y,exog=X_opt)
regress=regress.fit()
regress.summary()

X_opt=X[:,[0,1,3,4,5]]
regress=sm.OLS(endog=Y,exog=X_opt)
regress=regress.fit()
regress.summary()


X_opt=X[:,[0,3,4,5]]
regress=sm.OLS(endog=Y,exog=X_opt)
regress=regress.fit()
regress.summary()


X_opt=X[:,[0,3,5]]
regress=sm.OLS(endog=Y,exog=X_opt)
regress=regress.fit()
regress.summary()
#VISUAL ANALYSIS
plt.scatter(X[:,2],Y)     #x is r&d spend and y is profit
sns.regplot(X[:,2],Y)
sns.regplot(mydata["R&D Spend"],mydata["Profit"])
sns.regplot(mydata["Administration"],mydata["Profit"])
sns.regplot(mydata["Marketing Spend"],mydata["Profit"])

sns.pairplot(mydata,kind='reg')
sns.pairplot(mydata,kind='reg',hue='State')


#FIT generalised method whenever applied on any object 4 any kind of dataset it will fit the object in the dataset
#transform converts the data
#fit transform fits and then transforms
