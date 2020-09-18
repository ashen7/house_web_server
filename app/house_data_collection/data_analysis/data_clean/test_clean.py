# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:45:52 2018

@author: ying.zhang01
"""

import os
import re
import csv
import pandas as pd

"""
1、读入数据.2、清理数据.3、写出数据.
"""
filename = "../../data/ershoufang.csv"
with open(filename, encoding="utf-8") as f:
    reader = csv.reader(f)
    context = [line for line in reader]

filename = "../../data/广州二手房清洗后数据.csv"
with open(filename,"w",encoding="utf-8",newline="") as f:
    writer = csv.writer(f, dialect='excel')
    for i,line in enumerate(context):
        line = [x.strip() for x in line]#去除每个数据项的空白符和换行符
        if line[0] == "id":
            line[3] += "(万元)"
            line[4] += "(元/平米)"
            writer.writerow(line)
            continue
        if len(line) != 25:
            print(i, len(line), line)
        
        #将杂乱的记录的数据项对齐
        if "别墅" in line:
            line_copy = line[:]
            line[8] = "null"
            line[9] = line_copy[8]
            line[10] = "null"
            line[11] = line_copy[9]
            line[12] = line_copy[10]
            line[13] = line_copy[11]
            line[14] = "null"
            line[15] = "null"
            line[16] = line_copy[13]
        if "商业办公类" in line:
            #正则表达式匹配
            result = re.match(r"\d{4}-\d{1,2}-\d{1,2}",line[17])
            if result is None:
                del line[17]
            result = re.match(r"\d{4}-\d{1,2}-\d{1,2}",line[17])
            if result is None:
                del line[17]
            result = re.match(r"\d{4}-\d{1,2}-\d{1,2}",line[17])
            if result is None:
                del line[17]
        if "车库" in line:
            line_copy = line[:]
            line[5] = "null"
            line[6] = line_copy[5]
            line[7] = "null"
            line[11] = line_copy[7]
        if "商水" in line or "民水" in line or "商电" in line:
            line_copy = line[:]
            line[16] = "null"
            line[17] = line[18]
            line[18] = line[19]
            line[19] = line[20]
            line[20] = line[21]
            line[21] = line[22]
            line[22] = line[23]
            line[23] = line[24]
            line[24] = line[25]
            del line[25]
        if "3.45元/m3" in line or "4.36元/m3" in line or "2.8元/m3" in line or "3.5元/m3" in line:
            line_copy = line[:]
            line[16] = "null"
            line[17] = line[18]
            line[18] = line[19]
            line[19] = line[20]
            line[20] = line[21]
            line[21] = line[22]
            line[22] = line[23]
            line[23] = line[24]
            line[24] = line[25]
            del line[25]
        if len(line) == 26:
            line_copy = line[:]
            line[16] = "null"
            line[17] = line[18]
            line[18] = line[19]
            line[19] = line[20]
            line[20] = line[21]
            line[21] = line[22]
            line[22] = line[23]
            line[23] = line[24]
            line[24] = line[25]
            del line[25]
        if len(line) == 27:
            line_copy = line[:]
            line[16] = "null"
            line[17] = "null"
            line[18] = line[20]
            line[19] = line[21]
            line[20] = line[22]
            line[21] = line[23]
            line[22] = line[24]
            line[23] = line[25]
            line[24] = line[26]
            del line[25]
            # del line[26]

        try:
            #将总价数据项统一整理为整数    
            float_num = float(line[3])
            line[3] = str(int(float_num))
        
            #去除单价数据项单位
            line[4] = line[4].split("元")[0]
            
            #去除建筑面积数据项的单位
            if line[7] != "null" and line[7] != "暂无数据":
                line[7] = line[7].split("㎡")[0]
            
            #去除套内面积数据项的单位
            if line[9] != "null" and line[9] != "暂无数据":
                line[9] = line[9].split("㎡")[0]
            
            writer.writerow(line)
        except Exception as e:
            print("数据项转换失败!该记录未写入")


# house_data = pd.read_csv(filename, error_bad_lines=False)
# house_data.drop(house_data.columns[16], axis=1, inplace=True)
# print(house_data)
