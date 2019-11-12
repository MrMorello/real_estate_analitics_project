import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg



df_adv = pd.read_excel('tables\gipernn_adv.xlsx', index_col=0)
df_adv['mean_price_sqrm'] = df_adv['price'] / df_adv['total_square']
housing = df_adv.copy()

housing['build_year'] = housing['build_year'].replace('Уточнить', None)
housing['build_year'] = housing['build_year'].astype('float64', errors='ignore')
housing['current_floor'] = housing['current_floor'].replace('цоколь ', 0)
housing['current_floor'] = housing['current_floor'].astype('float64')
housing['ceiling_height'] = housing['ceiling_height'].replace('Уточнить', None)
housing['ceiling_height'] = housing['ceiling_height'].astype('float64')

california_img=mpimg.imread('images\california2.png')
ax = housing.plot(kind="scatter", x="longtitude", y="latitude", figsize=(10, 7),
                       s=3, label="Objects",
                       c=housing['mean_price_sqrm'], cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4)
plt.imshow(california_img, extent=[43.6, 44.1, 56.15, 56.4], alpha=0.8,
           cmap=plt.get_cmap("jet"))
plt.ylabel("latitude", fontsize=14)
plt.xlabel("longtitude", fontsize=14)
prices = housing["mean_price_sqrm"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
plt.legend(fontsize=16)
plt.savefig("images\colorized_map_plot", dpi=400)
housing.describe().to_excel('tables\describe_table.xlsx')
housing.hist(bins=50, figsize=(20,15))
plt.savefig('images\_02attribute_histogram_plots')

plt.show()

# Построение матрицы коррелиации и создание новых признаков
corr_matrix = housing.corr()
corr_matrix.to_excel('tables\corr_matrix.xlsx')
#Эксперементы с созданием новых признаков

from pandas.plotting import scatter_matrix
atributes = ["price", "total_square", "number_of_rooms", "build_year", "mean_price_sqrm"]
scatter_matrix(housing[atributes], figsize=(12, 8))
plt.savefig('images\_04_scatter_matrix_plots')

#Попробуем добавить новые признаки - обычно размер кухни имеет не последнее значение для покупателей, поэтому
# попробуем признак (площадь кухни к клощади квартиры), так же размер жилых комнат (жилая площадь к количеству комнат)
#Средний размер жилой комнаты
housing['avg_sqrm_living_room'] = housing['living_square'] / housing['number_of_rooms']
#Площадь кухни к площади квартиры
housing['kitchensqr_to_totalsqr'] = housing['kitchen_square'] / housing['total_square']
#Площадь кухни к жилой площади
housing['kitchensqr_to_livingsqr'] = housing['kitchen_square'] / housing['living_square']
corr_matrix = housing.corr()
corr_matrix.to_excel('tables\corr_matrix_experiment_features.xlsx')
#Видно что размер жилых комнат коррелирует с ценой прямопропорционально, остальные не коррелируют практически совсем
print("Done")




