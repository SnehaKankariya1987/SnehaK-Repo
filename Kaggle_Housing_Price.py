import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv(r'C:\Users\91998\OneDrive\Documents\Sneha Data Science and AI\Sneha_Git_Repo\SnehaK-Repo\Sneha_ML\Kaggle Real Estate Dataset.csv')
data=data.drop(['No'],axis=1)

X=data.drop(['Y house price of unit area'],axis=1)
Y=data[['Y house price of unit area']]

X.isnull().mean()
Y.isnull().mean()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_1=pd.DataFrame(imputer.fit_transform(X),index=X.index,columns=X.columns)

def outlier_cap(x):
    x=x.clip(lower=x.quantile(0.01))
    x=x.clip(upper=x.quantile(0.99))
    return(x)

X_1=X_1.apply(lambda x : outlier_cap(x))
# X_1 = outlier_cap(X_1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_1,Y,test_size=0.3,random_state=42)

from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train,y_train)

coeff_df=pd.DataFrame(X_1.columns)
coeff_df.columns=['features']
coeff_df["Coefficient Estimate"] = pd.Series(linreg.coef_[0])


linreg_pred_train=linreg.predict(X_train)
linreg_pred_test=linreg.predict(X_test)
linreg_pred_all=linreg.predict(X_1)
X_1['linreg_pred_prices']=pd.DataFrame(linreg_pred_all, index=X_1.index)

from sklearn.metrics import r2_score
r_sq_train=r2_score(linreg_pred_train,y_train)


from sklearn.metrics import r2_score
r_sq_test=r2_score(linreg_pred_test,y_test)


from sklearn.metrics import mean_squared_error

mse_train=mean_squared_error(linreg_pred_train,y_train)

mse_test=mean_squared_error(linreg_pred_test,y_test)

data_eval=pd.concat([X_1,Y],axis=1,join='inner')
data_eval['House_price_rank']=pd.qcut(data_eval['Y house price of unit area'].rank(method='first').values,5,duplicates='drop').codes+1

ax = sns.scatterplot( x="House_price_rank", y="Y house price of unit area", data=data_eval,color='Blue')
ax = sns.lineplot( x="House_price_rank", y="linreg_pred_prices", data=data_eval, color='Red')





