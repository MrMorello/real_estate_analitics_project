from encoding_trtest_split import test_set
from make_model import gradient_boosting
from sklearn.metrics import mean_squared_error
import numpy as np

test_data = test_set.drop(['price'], axis=1)
test_labels = test_set['price'].copy()

xgb_ts_pred = gradient_boosting.predict(test_data)
xgb_ch_mse = mean_squared_error(test_labels, xgb_ts_pred)
xgb_ch_rmse = np.sqrt(xgb_ch_mse)
print("\nXGB FINAL test score: {:.4f}\n".format(xgb_ch_rmse))

test_data.to_excel('tables\data_sample.xlsx')