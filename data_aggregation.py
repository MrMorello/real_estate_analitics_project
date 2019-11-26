import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


df_adv = pd.read_excel('tables\gipernn_adv.xlsx', index_col=0)
#Сразу создадим признак, который напрашивается сам собой - цена за кв. метр
# Но его потом нужно сбросить, просто посмотрим как влияет а так же для аналитики
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

# сбрасываем цену за метр
housing = housing.drop(['mean_price_sqrm'], axis=1)

        #Подсказки ментора, создаем новые переменные и нормализуем значения(от 0 до 1)
#Средний размер жилой комнаты
housing['avg_sqrm_living_room'] = housing['living_square'] / housing['number_of_rooms']
# % жилой пл. от общей
housing['living_to_total'] = housing['living_square'] / housing['total_square']
#пл. кухни к жилой
housing['kitchen_to_living'] = housing['kitchen_square'] / housing['living_square']
#пл. кухни к общей
housing['kitchen_to_total'] = housing['kitchen_square'] / housing['total_square']
#текущий этаж / к-во этажей
housing['curFlor_to_totFlor'] = housing['current_floor'] / housing['number_of_floors']
#преобразуем год постройки в возраст
housing['house_age'] = 2019 - housing['build_year']

corr_matrix = housing.corr()
corr_matrix.to_excel('tables\corr_matrix_experiment_features.xlsx')
    #Поспотрим на матрицу коррелиации
#Проверим важность признаков после обучения модели используя feature importance


#Реализация признака удаление от центра города. (Kremlin lat=56.328351, long=44.001842)

from geopy.distance import distance
kremlin_coordinates = (56.328351, 44.001842)
housing['distance'] = housing.apply(
    (lambda row: distance(
        (row['latitude'], row['longtitude']),
        (kremlin_coordinates)
    ).km),
    axis=1
)
housing.to_excel('tables\AllDataBeforeOneHotEnc.xlsx')
        #           Train_Test_Split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42, shuffle=True)
housing_train = train_set.drop(['price'], axis=1)
housing_labels = train_set['price'].copy()
# ПЕРЕНАЗНАЧЕНИЕ ПЕРЕМЕННОЙ - надо поменять имена !
print(train_set)    # НЕ ШАФЛЮТСЯ !  ! !!
            #Нормализуем данные
normalized_housing = housing_train.copy()
from sklearn.preprocessing import MinMaxScaler
data_for_scale = normalized_housing[['total_square', 'living_square', 'kitchen_square', 'ceiling_height', 'house_age', 'distance', 'avg_sqrm_living_room']].copy()
scaler = MinMaxScaler()
print(scaler.fit(data_for_scale))
scaled_data = scaler.transform(data_for_scale)
print(type(scaled_data))
normalized_housing = pd.DataFrame(scaled_data, columns=['total_square', 'living_square', 'kitchen_square', 'ceiling_height', 'house_age', 'distance', 'avg_sqrm_living_room'])
#normalized_housing.to_excel('tables\_NormalizedDataBeforeOneHotEnc.xlsx') #сначала нужно перевест в DataFrame
housing_prepared = normalized_housing.join(housing[['living_to_total', 'kitchen_to_living', 'kitchen_to_total', 'curFlor_to_totFlor']])
housing_prepared.to_excel('tables\_NormalizedDataBeforeOneHotEnc.xlsx')
# теперь: OneHotEncoder - Количество комнат, дайон, материал
#
housing_cat = housing[['district', 'material', 'number_of_rooms']].copy()
encoder = OneHotEncoder()
housing_cat = encoder.fit_transform(housing_cat).toarray()
merged_enc_cat1 = encoder.categories_[0]
merged_enc_cat2 = encoder.categories_[1]
merged_enc_cat3 = encoder.categories_[2]
merged_enc_cat = [*merged_enc_cat1, *merged_enc_cat2, *merged_enc_cat3]
encoded_df = pd.DataFrame(data=housing_cat, columns=merged_enc_cat)
final_df = housing_prepared.join(encoded_df)
final_df.to_excel('tables\_alldataOHE.xlsx')