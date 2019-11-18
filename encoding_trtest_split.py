import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from data_aggregation import housing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

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
# матрица коррелиации
corr_matrix_OHE = catDF_plus_housingDF.corr()
corr_matrix_OHE.to_excel('tables\Матрица корелиации после OHE.xlsx')


#Убираем выбросы из общего набора данных
catDF_plus_housingDF = catDF_plus_housingDF.drop(catDF_plus_housingDF[(catDF_plus_housingDF['total_square'] > 300) | (catDF_plus_housingDF['price'] > 40000000)
| (catDF_plus_housingDF['mean_price_sqrm'] > 150000) | (catDF_plus_housingDF['mean_price_sqrm'] < 35000)].index)


# Сбрасываем цену за кв метр - mean_price_sqrm
catDF_plus_housingDF = catDF_plus_housingDF.drop(['mean_price_sqrm'], axis=1)
# Разделение на Train_Test SPLIT
train_set, test_set = train_test_split(catDF_plus_housingDF, test_size=0.2, random_state=42)

#delete outliers
#train_set = train_set.drop(train_set[(train_set['total_square'] > 300) | (train_set['price'] > 40000000)].index) |

fig, ax = plt.subplots()
ax.scatter(x = train_set['total_square'], y = train_set['price'])
plt.ylabel('price', fontsize=13)
plt.xlabel('total_square', fontsize=13)
plt.show()
#cбрасываем целевую переменную в трениновочном наборе
housing_train = train_set.drop(['price'], axis=1)
housing_labels = train_set['price'].copy()



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

