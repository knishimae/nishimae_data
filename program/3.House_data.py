import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go 
import cufflinks as cf

py.init_notebook_mode(connected=True)
cf.go_offline()

house= pd.read_csv("../data/kc_house_data_1.csv")

#ドル→円
house["jp_M"] = house["price"]*100/10000
#散布図
plt.scatter(house["jp_M"], house["sqft_living"],s = 1, color = "g")
#x軸タイトル, y軸タイトルを設定
plt.xlabel("jp_M")
plt.ylabel("sqft_living")

cond_vc = house["condition"].value_counts()
plt.pie(cond_vc,autopct="%.1f%%",labels=cond_vc.keys())
plt.title("condition pie")
plt.show()

plt.bar(house["grade"].value_counts().keys(), height = house["grade"].value_counts())
plt.xlabel("grade")
plt.ylabel("Number of occurrences")

plt.hist(house["jp_M"], bins = 20 ,range =[0,30000], color = "red")
plt.xlabel("jp_M")
plt.ylabel("Number of occurrences")

plt.figure(figsize= (13,8))
plt.subplot(2,2,1)
plt.scatter(house["jp_M"], house["sqft_living"],s = 1, color = "g")
plt.xlabel("jp_M")
plt.ylabel("sqft_living")
plt.subplot(2,2,2)
cond_vc = house["condition"].value_counts()
plt.pie(cond_vc,autopct="%.1f%%",labels=cond_vc.keys())
plt.title("condition pie")
plt.subplot(2,2,3)
plt.bar(house["grade"].value_counts().keys(), height = house["grade"].value_counts())
plt.xlabel("grade")
plt.ylabel("Number of  occurrences")
plt.subplot(2,2,4)
plt.hist(house["jp_M"], bins = 20 ,range =[0,30000], color = "red")
plt.xlabel("jp_M")
plt.ylabel("Number of  occurrences")
plt.savefig("matplot.jpg")

house.groupby("yr_built").mean()["price"].plot()

fig,ax = plt.subplots(2,2,figsize=(13,8))
sns.pointplot("bedrooms","jp_M",data=house,hue="condition",ax=ax[0,0])

sns.boxplot("grade","sqft_living",data=house,ax=ax[0,1])

sns.swarmplot("grade","jp_M",data=house.loc[house["grade"]>9],ax=ax[1,0])

sns.violinplot(x="grade",y="bathrooms",data=house.loc[(house["grade"]>3) & (house["grade"]< 6)], hue="good/bad",split=True,ax=ax[1,1])house.groupby("sqft_living").mean()["jp_M"].iplot()

house.groupby("sqft_living").mean()["jp_M"].iplot()

