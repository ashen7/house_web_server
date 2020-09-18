import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
#用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

"""1、数据加载"""
#定义加载数据的文件名
filename = "../../data/广州二手房清洗后数据.csv"
#自定义数据的行列索引（行索引使用pd默认的，列索引使用自定义的）
names = [
        "id","communityName","areaName","total","unitPriceValue",
        "fwhx","szlc","jzmj","hxjg","tnmj",
        "jzlx","fwcx","jzjg","zxqk","thbl",
        "pbdt","cqnx","gpsj","jyqs","scjy",
        "fwyt","fwnx","cqss","dyxx","fbbj",
        ]

#自定义需要处理的缺失值标记列表
miss_value = ["null","暂无数据"]
#数据类型会自动转换
#使用自定义的列名，跳过文件中的头行，处理缺失值列表标记的缺失值
df = pd.read_csv(filename,skiprows=[0],names=names,na_values=miss_value)
# print(df.info())
# print(df)

"""2、数据运算"""
"""3、数据可视化呈现"""


"""广州各区域二手房平均单价"""
#数据分组、数据运算和聚合
groups_unitprice_area = df["unitPriceValue"].groupby(df["areaName"])
mean_unitprice = groups_unitprice_area.mean()
print(mean_unitprice)
mean_unitprice.index.name = "各区域名称"

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ax.set_xlabel(mean_unitprice.index.name)
ax.set_ylabel("单价(元/平米)")
ax.set_title("广州各区域二手房平均单价")
mean_unitprice.plot(kind="bar", color='g')
plt.savefig('../../../../static/images/广州各区域二手房平均单价.png')

"""广州各区域二手房平均建筑面积"""
#数据运算
groups_area_jzmj = df["jzmj"].groupby(df["areaName"])
mean_jzmj = groups_area_jzmj.mean()
print(mean_jzmj)
mean_jzmj.index.name = "各区域名称"

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ax.set_xlabel(mean_jzmj.index.name)
ax.set_ylabel("建筑面积(㎡)")
ax.set_title("广州各区域二手房平均建筑面积")
mean_jzmj.plot(kind="bar", color='g')
plt.savefig('../../../../static/images/广州各区域二手房平均建筑面积.png')

"""广州各区域平均单价和平均建筑面积"""
groups_unitprice_area = df["unitPriceValue"].groupby(df["areaName"])
mean_unitprice = groups_unitprice_area.mean()
print(mean_unitprice)
mean_unitprice.index.name = ""

groups_area_jzmj = df["jzmj"].groupby(df["areaName"])
mean_jzmj = groups_area_jzmj.mean()
print(mean_jzmj)
mean_jzmj.index.name = "各区域名称"

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.set_ylabel("单价(元/平米)")
ax1.set_title("广州各区域二手房平均单价")
ax2 = fig.add_subplot(2,1,2)
ax2.set_ylabel("建筑面积(㎡)")
ax2.set_title("广州各区域二手房平均建筑面积")
plt.subplots_adjust(hspace=0.4)
mean_unitprice.plot(kind="bar",ax=ax1)
mean_jzmj.plot(kind="bar",ax=ax2)
plt.savefig('../../../../static/images/广州各区域二手房平均单价和平均建筑面积.png')


"""广州各区域二手房房源数量"""
groups_area = df["id"].groupby(df["areaName"])
count_area = groups_area.count()
count_area.index.name = "各区域名称"
print("=="*40)

print(count_area)
exit(0)
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ax.set_ylabel("房源数量(套)")
ax.set_title("广州各区域二手房房源数量")
count_area.plot(kind="bar", color='g')
plt.savefig('../../../../static/images/广州各区域二手房房源数量.png')


"""广州二手房单价最高Top10"""
unitprice_top = df.sort_values(by="unitPriceValue",ascending=False)[:10]
print(unitprice_top)
unitprice_top.set_index(unitprice_top["communityName"],inplace=True)
unitprice_top.index.name = ""

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ax.set_ylabel("单价(元/平米)")
ax.set_title("广州二手房单价最高Top10")
unitprice_top["unitPriceValue"].plot(kind="bar")
plt.savefig('../../../../static/images/广州二手房单价最高Top10.png')

"""广州二手房房屋户型占比情况"""
count_fwhx = df['fwhx'].value_counts()[:10]
print(count_fwhx)
count_other_fwhx = pd.Series({"其他":df['fwhx'].value_counts()[10:].count()})
print(count_other_fwhx)
count_fwhx = count_fwhx.append(count_other_fwhx)
print(count_fwhx)
count_fwhx.index.name = ""
count_fwhx.name = ""

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_title("广州二手房房屋户型占比情况")
count_fwhx.plot(kind="pie",cmap=plt.cm.rainbow,autopct="%3.1f%%")
plt.savefig('../../../../static/images/广州二手房房屋户型占比情况.png')


"""广州二手房房屋装修占比情况"""
count_zxqk = df["zxqk"].value_counts()
print(count_zxqk)
count_zxqk.name = ""

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_title("广州二手房装修占比情况")
count_zxqk.plot(kind="pie",cmap=plt.cm.rainbow,autopct="%3.1f%%")
plt.savefig('../../../../static/images/广州二手房装修占比情况.png')

"""广州二手房建筑类型占比情况"""
count_jzlx = df["jzlx"].value_counts()
count_jzlx.name = ""

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_title("广州二手房建筑类型占比情况")
count_jzlx.plot(kind="pie",cmap=plt.cm.rainbow,autopct="%3.1f%%")
plt.savefig('../../../../static/images/广州二手房建筑类型占比情况.png')

"""广州二手房房屋朝向分布情况"""
count_fwcx = df["fwcx"].value_counts()[:15]
count_other_fwcx = pd.Series({"其他":df['fwcx'].value_counts()[15:].count()})
count_fwcx = count_fwcx.append(count_other_fwcx)

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)
ax.set_title("广州二手房房源朝向分布情况")
count_fwcx.plot(kind="bar")
plt.show()
plt.savefig('../../../../static/images/广州二手房房源朝向分布情况.png')