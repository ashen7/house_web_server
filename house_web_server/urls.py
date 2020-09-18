"""house_web_server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from app import views

# 用视图函数处理匹配到的url  匹配url的任务用的是路由来实现的 并且支持正则
urlpatterns = [
    url(r'^admin/', admin.site.urls),       #管理员页面
    url(r'^$', views.index),                #主页
    url(r'^index/$', views.index),          #主页
    url(r'^index/index', views.index),      #主页

    url(r'^login/$', views.login),          #登录
    url(r'^index/login', views.login),      #登录
    url(r'^register/$', views.register),    #注册
    url(r'^index/register', views.register),#注册
    url(r'^logout/$', views.logout),        #登出
    url(r'^index/logout', views.logout),    #登出
    url(r'^__logout/$', views.__logout),        #登出
    url(r'^index/__logout', views.__logout),    #登出
    url(r'^user_page/$', views.user_page),              #用户主页
    url(r'^index/user_page', views.user_page),          #用户主页
    url(r'^root_page/$', views.root_page),              #管理员主页
    url(r'^index/root_page', views.root_page),          #管理员主页
    url(r'^index/visual', views.visual),                #可视化页面
    url(r'^index/root_visual', views.root_visual),      #管理员可视化页面
    url(r'^index/predict', views.predict),              #二手房估价页面
    url(r'^index/root_predict', views.root_predict),    #管理员二手房估价页面
    url(r'^index/data_collect', views.data_collect),    #管理员二手房数据采集页面

    url(r'^captcha/', include('captcha.urls')),    #验证码

    # url(r'^app/', include('app.urls', namespace='app')),
]

urlpatterns += staticfiles_urlpatterns()