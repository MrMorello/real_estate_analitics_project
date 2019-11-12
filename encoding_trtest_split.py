import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from data_aggregation import housing
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 50)
#Преобразуем в категориальные признаки: район (district) и материал стен (material)
# и сохраняем в отельный DataFrame-"dist_mat_df"

housing_cat = housing[['district', 'material']].copy()
encoder = OneHotEncoder()
housing_cat = encoder.fit_transform(housing_cat).toarray()
merged_enc_cat1 = encoder.categories_[0]
merged_enc_cat2 = encoder.categories_[1]
merged_enc_cat = [*merged_enc_cat1, *merged_enc_cat2]
#print(merged_enc_cat)
dis_mat_df = pd.DataFrame(data=housing_cat, columns=merged_enc_cat)
#print(dis_mat_df.cumsum())
catDF_plus_housingDF = housing.join(dis_mat_df) # ОБЪЕДИНЕННЫЙ ДАТАФРЕЙМ С КАТЕГОР ПРИЗНАКАМИ (дропнуть типы object)

#Сбрасываем типы object: district, address, material

catDF_plus_housingDF = catDF_plus_housingDF.drop(['district', 'address', 'material'], axis=1)
print(catDF_plus_housingDF.info())

# Разделение на Train_Test SPLIT
train_set, test_set = train_test_split(catDF_plus_housingDF, test_size=0.2, random_state=42)
#cбрасываем целевую переменную в трениновочном наборе
housing_train = train_set.drop(['price'], axis=1)
housing_labels = train_set['price'].copy()

# Масштабирование признаков
from sklearn.preprocessing import scale
scaled_housing_train = scale(housing_train)

'''
1. данный блок вставить перед вырисовыванием графиков
стратифицированный набор:
from sklearn.model_selection import StratifiedShuffleSplit   /* page 85

2. После графиков
Поиск связей .corr() 
from pandas.plotting import scatter_matrix
3. Создание снтетических признаков и эксперименты с ними
4. Сброс целевой переменной (price)
5. Очистка данных:
5.1 
'''

