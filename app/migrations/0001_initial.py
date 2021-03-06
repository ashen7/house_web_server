# Generated by Django 2.2.12 on 2020-04-16 15:48

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='HouseData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('community', models.CharField(max_length=128)),
                ('region', models.CharField(max_length=128)),
                ('total_price', models.FloatField()),
                ('unit_price', models.FloatField()),
                ('shape', models.CharField(max_length=128)),
                ('floor', models.CharField(max_length=128)),
                ('area', models.CharField(max_length=128)),
                ('orientation', models.CharField(max_length=128)),
                ('fix_up', models.CharField(max_length=128)),
                ('lift', models.CharField(max_length=128)),
                ('type', models.CharField(max_length=128)),
                ('use', models.CharField(max_length=128)),
                ('years', models.CharField(max_length=128)),
                ('current_time', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'verbose_name': '二手房数据',
                'verbose_name_plural': '二手房数据',
                'ordering': ['current_time'],
            },
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=128, unique=True)),
                ('password', models.CharField(max_length=256)),
                ('email', models.EmailField(max_length=254, unique=True)),
                ('mobile_number', models.CharField(default='', max_length=11, unique=True)),
                ('sex', models.CharField(choices=[('male', '男'), ('female', '女')], default='男', max_length=32)),
                ('current_time', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'verbose_name': '用户',
                'verbose_name_plural': '用户',
                'ordering': ['current_time'],
            },
        ),
    ]
