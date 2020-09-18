# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:31:53 2018

@author: zhangying
"""

import random
user_agent =[
            "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; InfoPath.2; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; 360SE) ",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0) ",
            "Mozilla/5.0 (Windows NT 5.1; zh-CN; rv:1.9.1.3) Gecko/20100101 Firefox/8.0",
            "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
            "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
            "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; TencentTraveler 4.0; .NET CLR 2.0.50727)",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36"
            ]

headers = {
        "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Encoding":"gzip, deflate, br",
        "Accept-Language":"zh-CN,zh;q=0.9",
        "Connection":"keep-alive",
        "Host":"nj.lianjia.com",
        "User-Agent":random.choice(user_agent)
        }
print(headers)

try:
    a = 5/0
except Exception as e:
    print(repr(e))


a='1,东方兰园,栖霞,399,32178,3室2厅1厨2卫,低楼层 (共21层),124,暂无数据,暂无数据,板楼,东 南 北,钢混结构,毛坯,两梯四户,有,70年,2018-03-31,商品房,暂无数据,普通住宅,暂无数据,共有,有抵押 140万元 交通银行 客户偿还,未上传房本照片'
ershoufang_data=a.split(',')
print(ershoufang_data)


import pymysql
import csv
import pandas as pd

house_data_file = "../../data/广州二手房清洗后数据.csv"
miss_value = ["null", "暂无数据"]
house_data = pd.read_csv(house_data_file, header=0, na_values=miss_value)

use_col_index_list = [1, 2, 3, 4, 5, 6, 7, 11, 13, 15, 18, 20, 21]
nouse_col_index_list = list()
for i in range(house_data.shape[1]):
    if i not in use_col_index_list:
        nouse_col_index_list.append(i)

house_data.drop(house_data.columns[nouse_col_index_list], axis=1, inplace=True)
print(house_data)
# exit(0)

# 获取connection连接 对象
db = pymysql.connect(host='localhost', port=3306, user='root', password='986300260',
                     database='house_data_db', charset='utf8')
# 获取游标对象 Cursor
cursor = db.cursor()
# insert into app_housedata values('翻斗小区', '3室2厅1卫', '28(高50层)', '150.0', '南', '精装', '有', '商品房', '普通住宅', '满二年');

i = 1
for row in range(house_data.shape[0]):
    data = list()
    flag = False
    for col in range(house_data.shape[1]):
        if house_data.iloc[row, col] != house_data.iloc[row, col]:
            print(house_data.iloc[row, col])
            if col in [2, 3]:
                flag = True
                break
            else:
                data.append("null")
                continue
        data.append(house_data.iloc[row, col])
    if flag:
        continue
    print(data)
    sql = "INSERT INTO app_housedata VALUES({}, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());"
    sql = sql.format(i)
    cursor.execute(sql, (data[0],
                         data[1],
                         float(data[2]),
                         float(data[3]),
                         data[4],
                         data[5],
                         str(data[6]),
                         data[7],
                         data[8],
                         data[9],
                         data[10],
                         data[11],
                         data[12]))
    db.commit()
    i += 1
cursor.close()
db.close()

