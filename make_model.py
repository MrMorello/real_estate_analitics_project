from data_aggregation import normalized_housing_test_X, housing_test_y, normalized_housing_train_X, housing_train_y, train_set, test_set
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
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
"""


# KNN
knn_reg = KNeighborsRegressor(n_neighbors=10)
knn_reg.fit(X=train_X, y=train_y)
knn_pred = knn_reg.predict(train_X)
temp_train_set_knn = train_set.copy()
temp_train_set_knn['predictions'] = knn_pred
#temp_train_set_knn.to_excel('tables\_test_set_redundantAnalisysKNN.xlsx')


#LinearRegressionT
e_net = LinearRegression(normalize=True)
e_net.fit(X=train_X, y=train_y)
e_net_predictions = e_net.predict(train_X)
temp_train_set_enet = train_set.copy()
temp_train_set_enet['predictions'] = e_net_predictions
temp_train_set_enet.to_excel('tables\_test_set_redundantAnalisysLinReg.xlsx')



#XGBoost
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', num_leaves=6,
                              learning_rate=0.07, n_estimators=820,
                              max_bin = 80, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=10, bagging_seed=9,
                              min_data_in_leaf =7, min_sum_hessian_in_leaf = 11)
model_xgb.fit(X=train_X, y=train_y)
print("XGB scores: ")
print(np.mean(cross_val_score(model_xgb, train_X, train_y, cv=5)))
xgb_predictions = model_xgb.predict(test_X)
#print(len(xgb_predictions))
#print(type(xgb_predictions))
#print(xgb_predictions)
#print(housing_test_y)
estimete_df = pd.DataFrame(test_y, columns=['price']).copy()
estimete_df['predictions'] = xgb_predictions
filename = 'finalized_model.sav'
pickle.dump(model_xgb, open(filename, 'wb'))


random_forest = RandomForestRegressor(n_estimators=200, random_state=42)
random_forest.fit(X=train_X, y=train_y)
print("Random Forest scores: ")
print(np.mean(cross_val_score(random_forest, train_X, train_y, cv=5)))
#print(random_forest.feature_importances_)
#   REDUNDANT ANALISYS
rf_predictions = random_forest.predict(train_X)
#estimate_rf = pd.DataFrame(train_y, columns=['price']).copy()
#estimate_rf['predictions'] = rf_predictions
temp_train_set = train_set.copy()
temp_train_set['predictions'] = rf_predictions
temp_train_set.to_excel('tables\_train_set_redundantAnalisys.xlsx')

# TEST SET check
rf_pred_test_set = random_forest.predict(test_X)
temp_test_set = test_set.copy()
temp_test_set['predictions'] = rf_pred_test_set
temp_test_set.to_excel('tables\_test_set_redundantAnalisysRF.xlsx')

XGB_pred_test_set = model_xgb.predict(test_X)
temp_test_set_XGB = test_set.copy()
temp_test_set_XGB['predictions'] = XGB_pred_test_set
temp_test_set_XGB.to_excel('tables\_test_set_redundantAnalisysXGB.xlsx')

etimation_XGB = pd.read_excel('tables\estimate_XGB.xlsx', index_col=0)
print(etimation_XGB.describe())
print('Done')