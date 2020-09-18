# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 21:33:42 2018

@author: ying.zhang01
"""

from wordcloud import WordCloud
import jieba
from scipy.misc import imread

"""南京二手房数据词云"""
#基础配置数据
filename = "../../data/广州二手房清洗后数据.csv"
backpicture = "../resources/house2.jpg"
savepicture = "../../../../static/images/广州二手房数据词云.png"
fontpath = "../resources/simhei.ttf"
stopwords = ["null","暂无","数据","上传","照片","房本"]

#读入数据文件
comment_text = open(filename,encoding="utf-8").read()  
# 读取背景图片
color_mask = imread(backpicture) 
  
#结巴分词,同时剔除掉不需要的词汇    
ershoufang_words = jieba.cut(comment_text)
ershoufang_words = [word for word in ershoufang_words if word not in stopwords]
# print(ershoufang_words)
cut_text = " ".join(ershoufang_words) 
print(cut_text)

#设置词云格式
cloud = WordCloud(
    #设置字体，不指定就会出现乱码
    font_path=fontpath,
    #设置背景色
    background_color='white',
    #词云形状
    mask=color_mask,
    #允许最大词汇
    max_words=2000,
    #最大号字体
    max_font_size=60
   )
# 产生词云
word_cloud = cloud.generate(cut_text) 
#保存图片
word_cloud.to_file(savepicture) 