from django.contrib import admin
from . import models

# Register your models here.

#admin中注册模型
admin.site.register(models.User)
admin.site.register(models.HouseData)

admin.AdminSite.site_header = 'admin管理系统'
admin.AdminSite.site_title = 'django管理后台'