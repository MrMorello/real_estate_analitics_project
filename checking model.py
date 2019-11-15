from encoding_trtest_split import test_set
from make_model import model_xgb, random_forest, gradient_boosting
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

test_data = test_set.drop(['price'], axis=1)
test_labels = test_set['price'].copy()

xgb_ts_pred = random_forest.predict(test_data)
xgb_ch_mse = mean_squared_error(test_labels, xgb_ts_pred)
xgb_ch_rmse = np.sqrt(xgb_ch_mse)
print("\nXGB FINAL test score: {:.4f}\n".format(xgb_ch_rmse))
test_set.to_excel('tables\Test_set.xlsx')

check_sample = pd.read_excel('tables\data_sample.xlsx')
check_sample_label = check_sample['price'].copy()
check_sample_data = check_sample.drop(['price', 'Unnamed: 0'], axis=1)
predict = random_forest.predict(check_sample_data)
print(predict)
print(list(check_sample_label))