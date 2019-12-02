from data_aggregation import normalized_housing_test_X, housing_test_y, normalized_housing_train_X, housing_train_y
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import scale, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
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

scaled_housing_train = scale(housing_train)
#XGBoost
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_xgb.fit(X=housing_train, y=housing_labels)
xgb_predictions = model_xgb.predict(housing_train)
xgb_mse = mean_squared_error(housing_labels, xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)
print("\nXGB score: {:.4f}\n".format(xgb_rmse))

k_neighbors = KNeighborsRegressor()
k_neighbors.fit(X=housing_train, y=housing_labels)
kn_predictions = k_neighbors.predict(housing_train)
kn_mse = mean_squared_error(housing_labels, kn_predictions)
kn_rmse = np.sqrt(kn_mse)
print("Kneighbors Без scale " + str(kn_rmse))
k_neighbors.fit(X=scaled_housing_train, y=housing_labels)
kn_predictions_s = k_neighbors.predict(scaled_housing_train)
kn_mse_s = mean_squared_error(housing_labels, kn_predictions)
kn_rmse_s = np.sqrt(kn_mse_s)
print("Kneighbors Используем scale " + str(kn_rmse_s))

gradient_boosting = GradientBoostingRegressor()
gradient_boosting.fit(X=housing_train, y=housing_labels)
gb_predictions = gradient_boosting.predict(housing_train)
gb_mse = mean_squared_error(housing_labels, gb_predictions)
gb_rmse = np.sqrt(gb_mse)
print("Gradient_Bossting RMSE = " + str(gb_rmse))

random_forest = RandomForestRegressor()
random_forest.fit(X=housing_train, y=housing_labels)
rf_predictions = random_forest.predict(housing_train)
rf_mse = mean_squared_error(housing_labels, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
print('Random forest RMSE '+ str(rf_rmse))

lin_reg = LinearRegression()
lin_reg.fit(X=housing_train, y=housing_labels)
housing_predictions = lin_reg.predict(housing_train)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('Linear Regression RMSE '+ str(lin_rmse))

print("MAE is:")
rf_mae = mean_absolute_error(housing_labels, rf_predictions)
print('Random_forest MAE '+str(rf_mae))
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print('Linear_regression MAE '+str(lin_mae))
kn_mae = mean_absolute_error(housing_labels, kn_predictions)
kn_mae_s = mean_absolute_error(housing_labels, kn_predictions_s)
print('kn_MAE without SCALE ' + str(kn_mae))
print('kn_MAE with SCALE ' + str(kn_mae_s))
#Получается что на этом этапе XGBoost самый многообещающий, нормализация не нужна https://stackoverflow.com/questions/8961586/do-i-need-to-normalize-or-scale-data-for-randomforest-r-package/8962851

#применяем кросс валидацию

