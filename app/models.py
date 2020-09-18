from django.db import models

# Create your models here.

# 用户表
class User(models.Model):
    # 类名对应一个数据表 类属性对应一个数据表中的字段
    gender = (
        ('male', '男'),
        ('female', '女')
    )
    name = models.CharField(max_length=128, unique=True)  #唯一
    password = models.CharField(max_length=256)
    email = models.EmailField(unique=True)                #唯一
    mobile_number = models.CharField(max_length=11, unique=True, default="")   #唯一
    sex = models.CharField(max_length=32, choices=gender, default='男')
    current_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    # 用户按创建时间的反序排列 最近的最先显示
    class Meta:
        ordering = ['current_time']
        verbose_name = '用户'
        verbose_name_plural = '用户'

# 二手房数据
class HouseData(models.Model):
    # 类名对应一个数据表 类属性对应一个数据表中的字段
    community = models.CharField(max_length=128)
    region = models.CharField(max_length=128)
    total_price = models.FloatField()
    unit_price = models.FloatField()
    shape = models.CharField(max_length=128)
    floor = models.CharField(max_length=128)
    area = models.CharField(max_length=128)
    orientation = models.CharField(max_length=128)
    fix_up = models.CharField(max_length=128)
    lift_scale = models.CharField(max_length=128, null=True)
    lift = models.CharField(max_length=128)
    type = models.CharField(max_length=128)
    use = models.CharField(max_length=128)
    years = models.CharField(max_length=128)
    current_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.community

    class Meta:
        ordering = ['current_time']
        verbose_name = '二手房数据'
        verbose_name_plural = '二手房数据'
