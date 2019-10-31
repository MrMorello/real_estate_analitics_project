import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from data_aggregation import housing

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




# Разделение на Train_Test SPLIT

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

