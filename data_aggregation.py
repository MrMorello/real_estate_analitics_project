import pandas as pd
#import numpy as np

df_adv = pd.read_excel('D:\_Machine Learning\gipernn_adv.xlsx', index_col=0)
df_geo = pd.read_excel('D:\_Machine Learning\housing_DB_lat_long.xlsx', index_col=0)

adv_np = df_adv.to_numpy()
geo_np = df_geo.to_numpy()
adv_longtitude = []
adv_latitude = []

for index, row in df_adv.iterrows():
    row['address'] = row['address'].replace(' д.', '')
    row['address'] = row['address'].split(" ", maxsplit=1)[1:][0]
    print(row['address'])

for index_adv, row_adv in df_adv.iterrows():
    for index_geo, row_geo in df_geo.iterrows():
        if row_adv['address'] == row_geo['addres']:
            adv_longtitude.append(row_geo['longtitude'])
            adv_latitude.append(row_geo['latitude'])

        else:
            pass
            
        print(len(adv_latitude))
        print(len(adv_longtitude))




print(df_adv.info())
print(df_geo.info())