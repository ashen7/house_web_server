# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:35:07 2018

@author: zhangying
"""

import time
import random
import pymysql

from app.house_data_collection.data_collection.lianjia.url_manager import UrlManager
from app.house_data_collection.data_collection.lianjia.log import MyLog
from app.house_data_collection.data_collection.lianjia.html_downloader import HtmlDownloader
from app.house_data_collection.data_collection.lianjia.html_parser import HtmlParser
from app.house_data_collection.data_collection.lianjia.html_outputer import HtmlOutputer
# from url_manager import UrlManager
# from log import MyLog
# from html_downloader import HtmlDownloader
# from html_parser import HtmlParser
# from html_outputer import HtmlOutputer

class SpiderMain():
    """爬虫程序主模块"""
    def __init__(self):
        """构造函数，初始化属性"""
        self.urls = UrlManager()
        self.log = MyLog("spider_main","logs")
        self.downloader = HtmlDownloader()
        self.parser = HtmlParser()
        self.outputer = HtmlOutputer()

    def craw(self,root_url):
        """爬虫入口函数"""
        # 获取connection连接 对象
        db = pymysql.connect(host='localhost', port=3306, user='root', password='986300260',
                             database='house_data_db', charset='utf8')
        # 获取游标对象 Cursor
        cursor = db.cursor()
        use_col_index_list = [1, 5, 6, 7, 11, 13, 15, 18, 20, 21]

        # 广州市二手房 1051*30=31530
        areas = {
                "tianhe":1, "yuexiu":1, "liwan":100,
                "haizhu":100,"panyu":100, "baiyun":100,
                "huangpugz":100, "conghua":30,"zengcheng":100,
                "huadou":100, "nansha":50, "nanhai":20,
                "shunde":1
                }

        filename = "app/house_data_collection/data/urls.txt"
        write_urls = "app/house_data_collection/data/urls2.txt"
        try:
            urls_file = open(filename, 'r', encoding='utf-8')
            print("成功打开urls文件!")
        except Exception as e:
            print(e)
            #1、抓取所有二手房详情界面链接，并将所有连接放入URL管理模块
            for area, pg_sum in areas.items():
                for num in range(1, pg_sum+1):
                    #1.1 拼接页面地址: https://gz.lianjia.com/ershoufang/tianhe/pg2/
                    pg_url = root_url + area + "/pg" + str(num) + "/"
                    self.log.logger.info("1.1 拼接页面地址：" + pg_url)
                    print("1.1 拼接页面地址：" + pg_url)
                    #1.2 启动下载器,下载页面.
                    try:
                        html_content = self.downloader.download(pg_url)
                    except Exception as e:
                        self.log.logger.error("1.2 下载页面出现异常:" + repr(e))
                        time.sleep(60*20)
                    else:
                        #1.3 解析PG页面，获得二手房详情页面的链接,并将所有链接放入URL管理模块
                        try:
                            ershoufang_urls = self.parser.get_erhoufang_urls(html_content)
                        except Exception as e:
                            self.log.logger.error("1.3 页面解析出现异常:" + repr(e))
                        else:
                            self.urls.add_new_urls(ershoufang_urls)
                            #暂停0~3秒的整数秒，时间区间：[0,3]
                            time.sleep(random.randint(0,3))

            urls = self.urls.get_urls()
            print(len(urls))
            write_urls_file = open(write_urls, 'w', encoding='utf-8')
            for url in urls:
                write_urls_file.writelines(url + '\n')
            write_urls_file.close()


        # time.sleep(60*10)
        urls = [url.strip() for url in urls_file.readlines()]
        self.urls.add_new_urls(urls)
        print(len(self.urls.get_urls()))

        #2、解析二手房具体细心页面
        id = 1
        stop = 1
        while self.urls.has_new_url():
            #2.1 获取url
            try:
                detail_url = self.urls.get_new_url()
                self.log.logger.info("2.1 二手房页面地址：" + detail_url)
                print("2.1 二手房页面地址：" + detail_url)
            except Exception as e:
                print("2.1 拼接地址出现异常")
                self.log.logger.error("2.1 拼接地址出现异常:" + detail_url)
            
            #2.2 下载页面
            try:
                detail_html = self.downloader.download(detail_url)
            except Exception as e:
                self.log.logger.error("2.2 下载页面出现异常:" + repr(e))
                self.urls.add_new_url(detail_url)
                time.sleep(60*10)
            else:
                #2.3 解析页面
                try:
                    ershoufang_data = self.parser.get_ershoufang_data(detail_html, id)
                except Exception as e:
                    self.log.logger.error("2.3 解析页面出现异常:" + repr(e))
                else:
                    #2.4 输出数据
                    try:
                        self.outputer.collect_data(ershoufang_data)
                        # 存入数据库 sql语句
                        sql = "INSERT INTO app_housedata VALUES({}, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW());"
                        sql = sql.format(id)
                        # cursor.execute(sql, (ershoufang_data[use_col_index_list[0]],
                        #                      ershoufang_data[use_col_index_list[1]],
                        #                      ershoufang_data[use_col_index_list[2]],
                        #                      ershoufang_data[use_col_index_list[3]],
                        #                      ershoufang_data[use_col_index_list[4]],
                        #                      ershoufang_data[use_col_index_list[5]],
                        #                      ershoufang_data[use_col_index_list[6]],
                        #                      ershoufang_data[use_col_index_list[7]],
                        #                      ershoufang_data[use_col_index_list[8]],
                        #                      ershoufang_data[use_col_index_list[9]]))
                        # db.commit()
                    except Exception as e:
                        self.log.logger.error("2.4 输出数据出现异常:" + repr(e))
                    else:
                        print(id)
                        id = id + 1
                        stop = stop + 1
                        #暂停0~3秒的整数秒，时间区间：[0,3]
                        time.sleep(random.randint(0,3))
                        if stop == 2500:
                            stop = 1;
                            time.sleep(60*10)
        cursor.close()
        db.close()
            
def data_collection_api():
    # 设定爬虫入口URL
    root_url = "https://gz.lianjia.com/ershoufang/"
    # 初始化爬虫对象
    obj_spider = SpiderMain()
    # 启动爬虫
    obj_spider.craw(root_url)

if __name__ == "__main__":
    #设定爬虫入口URL
    root_url = "https://gz.lianjia.com/ershoufang/"
    #初始化爬虫对象
    obj_spider = SpiderMain()
    #启动爬虫
    obj_spider.craw(root_url)