#ライブラリのインポート
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

#データセットのインポート
train=pd.read_csv("../data/train.csv")
test=pd.read_csv("../data/test.csv")
data=[train,test]

#生き残った数と無くなった数を棒グラフで表示
s_rate = train["Survived"].mean()
s_num = train["Survived"].value_counts()

#プロット
fig,ax=plt.subplots(1,2,figsize=(8,4))
ax[0].pie(s_num,labels=["Deceased","Survived"],autopct="%.1f%%")
ax[0].set_title("Ratio")

sns.countplot(train["Survived"],ax=ax[1])
ax[1].set_title("Number")

#男女で生き残った数と無くなった数をグラフ表示
print(train.groupby(["Sex","Survived"])["Survived"].count())
train.groupby(["Sex","Survived"])["Survived"].count().plot.bar()

#高いFareを払った人の順番（降順）に並べかえた名簿の上から順に一人一人の年齢をインタラクティブなグラフに表示
train.sort_values("Fare",ascending=False).reset_index()["Age"].iplot()

#名前は敬称だけ抜き取る
import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

#"Title"という列を作り、その中に全員分の敬称を収録する
for df in [train,test]:
    df['Title'] = df['Name'].apply(get_title)
    
df['Title']
pirnt(train)

train['Title'].value_counts()

#Mr, Miss, Others(その他)におきかえ
for df in data:
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')
    df['Title']=df['Title'].replace(["Mlle","Ms","Mrs"],"Miss")
    df['Title']=df['Title'].replace(["Mme","Master","Mrs"],"Mr")
train['Title'].value_counts()

#敬称別での年齢の平均をとって、欠損値を埋める
for df in [train,test]:
    for title in train["Title"].unique():
        df.loc[(df.Age.isnull())&(df.Title==title),"Age"] = df.loc[df.Title==title,'Age'].mean()
sns.boxplot(train["Age"])

def cotegorize_age(age):
    if age>59:
        return 4
    if age>37:
        return 3
    if age>30:
        return 2
    if age>22:
        return 1
    return 0

def cotegorize_fare(fare):
    if fare>31:
        return 3
    if fare>15:
        return 2
    if fare>8:
        return 1
    return 0

for df in [train,test]:
    df["Age_Band"] = df["Age"].apply(cotegorize_age)
    df["Fare_Band"] = df["Fare"].apply(cotegorize_fare)

#敬称(Title)と支払った運賃(Fare)の関係
sns.swarmplot("Title","Fare",data=train)

#運賃のクラス(Fare_band)・性別(Sex)と生存率の関係
sns.factorplot("Fare_Band","Survived",data=train,col="Title",hue="Sex")

# 上の6つのcolumn名をリストに。
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin','Age','Fare']
train = train.drop(drop_columns, axis = 1)
test  = test.drop(drop_columns, axis = 1)

data=[train,test]

for df in data:
    # 性別を数字でおきかえ
    df.loc[df['Sex']=="female", "Sex"]=0
    df.loc[df['Sex']=='male','Sex']=1
    
    # 敬称を数字で置き換え
    df.loc[df['Title']=='Mr', 'Title']=0
    df.loc[df['Title']=='Miss', 'Title']=1
    df.loc[df['Title']=='Mrs', 'Title']=2
    df.loc[df['Title']=='Master', 'Title']=3
    df.loc[df['Title']=='Others', 'Title']=4
    
    # 乗船した港３種類を数字でおきかえ
    df.loc[df['Embarked']=='S', 'Embarked']=0
    df.loc[df['Embarked']=='C', 'Embarked']=1
    df.loc[df['Embarked']=='Q', 'Embarked']=2

print(train)

