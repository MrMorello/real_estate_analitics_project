from data_aggregation import normalized_housing_test_X, housing_test_y, normalized_housing_train_X, housing_train_y, test_set
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
import pandas as pd
import pickle

train_X = normalized_housing_train_X.copy()
train_y = housing_train_y.copy()
test_X = normalized_housing_test_X.copy()
test_y = housing_test_y.copy()

"""
#Example from KAGGLE
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(housing_train.values)
    rmse = np.sqrt(-cross_val_score(model, housing_train.values, housing_labels, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
elscore = rmsle_cv(ENet)
print("ElasticNet score: {:.4f}\n".format(elscore.mean()))

#Elastic NET
e_net = ElasticNet()
e_net.fit(X=housing_train, y=housing_labels)
e_net_predictions = e_net.predict(housing_train)
e_net_mse = mean_squared_error(housing_labels, e_net_predictions)
e_net_rmse = np.sqrt(e_net_mse)
print("\nElastic Net score: {:.4f}\n".format(e_net_rmse))
"""
e_net = ElasticNet()
e_net.fit(X=train_X, y=train_y)
e_net_predictions = e_net.predict(train_X)
e_net_mae = mean_absolute_error(train_y, e_net_predictions)
print("\nElastic Net score: {:.4f}\n".format(e_net_mae))

#XGBoost
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', num_leaves=6,
                              learning_rate=0.06, n_estimators=820,
                              max_bin = 80, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_xgb.fit(X=train_X, y=train_y)
print("XGB scores: ")
print(np.mean(cross_val_score(model_xgb, train_X, train_y, cv=5)))
xgb_predictions = model_xgb.predict(train_X)
print(mean_absolute_error(train_y, xgb_predictions))
xgb_predictions = model_xgb.predict(test_X)
print(len(xgb_predictions))
print(type(xgb_predictions))
print(xgb_predictions)
print(housing_test_y)
estimete_df = pd.DataFrame(test_y, columns=['price']).copy()
estimete_df['predictions'] = xgb_predictions
filename = 'finalized_model_clear.sav'
pickle.dump(model_xgb, open(filename, 'wb'))




random_forest = RandomForestRegressor(n_estimators=90, max_depth=6, random_state=42)
random_forest.fit(X=train_X, y=train_y)
print("Random Forest scores: ")
print(np.mean(cross_val_score(random_forest, train_X, train_y, cv=5)))
print(random_forest.feature_importances_)
