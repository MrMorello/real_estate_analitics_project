from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from data_aggregation import housing
import pandas as pd

#Преобразуем в категориальные признаки: район (district) и материал стен (material)

housing_cat = housing[['district', 'material']].copy()
le = LabelEncoder()
housing_cat['district'] = le.fit_transform(housing_cat['district'])
housing_cat['material'] = le.fit_transform(housing_cat['material'])
print(housing_cat)
print(housing_cat['district'].value_counts())
print(housing_cat['material'].value_counts())
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

