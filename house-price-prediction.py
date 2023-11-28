from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow import keras
from xgboost import XGBRegressor


import numpy as np
import pandas as pd
import math


def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def apply_pca(X, standardize=True):
    # Standardize
    X = X.copy()
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


def xgboost_train_run(x_train, y_train, x_test, y_test):
    model = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
    model.fit(x_train, y_train)
    p_train = model.predict(x_test)
    mse = metrics.mean_squared_error(np.log(p_train), np.log(y_test))
    rmse = math.sqrt(mse)
    print(f"rmse {rmse}")   
    return model, p_train



def preprocess(data):
    
    for colname in data.select_dtypes("object"):
        data[colname], _ = data[colname].factorize()

    

    data['Functional'] = data['Functional'].fillna('Typ')
    data['Electrical'] = data['Electrical'].fillna("SBrkr")
    data['KitchenQual'] = data['KitchenQual'].fillna("TA")
    data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
    data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
    data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])

    data["PoolQC"] = data["PoolQC"].fillna("None")

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        data[col] = data[col].fillna(0)
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        data[col] = data[col].fillna('None')
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        data[col] = data[col].fillna('None')

    data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    data.fillna(0, inplace=True)
    data["LivLotRatio"] = data["GrLivArea"] / data["LotArea"]
    data["Spaciousness"] = (data["1stFlrSF"] + data["2ndFlrSF"])/data["TotRmsAbvGrd"]
    data["TotalOutsideSF"] = data["WoodDeckSF"]+ data["OpenPorchSF"]+ data["EnclosedPorch"] + data["ScreenPorch"]
    components = [ "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
                "ScreenPorch"]
    data["PorchTypes"] = data[components].gt(0).sum(axis=1)

    data["MedNhbdArea"] = (data.groupby("Neighborhood")["GrLivArea"].transform("median"))   

    features_pca = [
        "GarageArea",
        "YearRemodAdd",
        "TotalBsmtSF",
        "GrLivArea",
        "GarageCars",
        "YearBuilt"
    ]


    pca, X_pca, loadings = apply_pca(data.loc[:, features_pca])
    data = data.join(X_pca)




    features = ['LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','GrLivArea']
    
    # Standardize
    X_scaled = data.loc[:, features].copy()
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=10, random_state=0)
    data["Cluster"] = kmeans.fit_predict(X_scaled)
    return data


train_data = pd.read_csv('train.csv')
y_train = train_data.SalePrice
train_data = train_data.drop(['SalePrice'], axis=1)
train_data = preprocess(train_data)

(x_train, x_test, y_train, y_test) = train_test_split(train_data, y_train, test_size = .25)

#model, p_train = nn_train_run(x_train, y_train, x_test, y_test)
model, p_train = xgboost_train_run(x_train, y_train, x_test, y_test)






test_data = pd.read_csv('test.csv')

test_data = preprocess(test_data)

predictions = model.predict(test_data)

final_pd = pd.DataFrame(test_data.Id, columns=['Id', 'SalePrice'])
final_pd.SalePrice = predictions
best_pd = pd.read_csv('predictions_xgboost_new.csv')
mse = metrics.mean_squared_error(np.log(predictions), np.log(best_pd.SalePrice))
rmse = math.sqrt(mse)
print(f"TEST rmse {rmse}") 
final_pd.to_csv('predictions_xgboost_latest.csv', index=False)

# score 0.15444 and score 0.15236 gave a rmse of 0.099