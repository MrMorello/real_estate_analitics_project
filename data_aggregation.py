import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

df_adv = pd.read_excel('D:\_Machine Learning\gipernn_adv.xlsx', index_col=0)
df_adv['mean_price_sqrm'] = df_adv['price'] / df_adv['total_square']
housing = df_adv.copy()



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
#cbar = plt.colorbar()
#cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
#cbar.set_label('mean price sqrm', fontsize=16)

plt.legend(fontsize=16)
plt.savefig("images\colorized_map_plot", dpi=400)
plt.show()

