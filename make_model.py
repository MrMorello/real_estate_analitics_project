from encoding_trtest_split import housing_train, housing_labels,scaled_housing_train
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
lin_reg = LinearRegression()
lin_reg.fit(X=housing_train, y=housing_labels)

housing_predictions = lin_reg.predict(housing_train)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)