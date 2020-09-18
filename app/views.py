from django.shortcuts import render, redirect
from django.http import HttpResponse
from .form import UserForm, RegisterForm, HouseDataForm
from . import models
import hashlib
import time
from multiprocessing import Process

from .house_predict.predict_house_price_test import predict_house_price_api
from .house_data_collection.data_collection.lianjia.spider_main import data_collection_api
from .task import task_api

# Create your views here.

# 视图函数 服务器返回http response
def index(request):
    # 1. 去模型Models里取数据
    # 2. 将数据传递给模板  模板渲染页面 将渲染好的页面返回浏览器
    return render(request, 'index.html')

def hash_coder(str, salt='website'):  #加salt
    hash_code = hashlib.sha256()
    str += salt
    hash_code.update(str.encode())
    return hash_code.hexdigest()

# 登录
def login(request):
    # 不允许重复登录
    if request.session.get('is_login', None):
        if request.session.get('user_name', None) == "admin":
            return redirect("/root_page/")
        else:
            return redirect("/user_page/")

    if request.method == "POST":
        login_form = UserForm(request.POST)
        message = "Please check the content！"
        if login_form.is_valid():
            username = login_form.cleaned_data['username']
            password = login_form.cleaned_data['password']
            try:
                user = models.User.objects.get(name=username)
                if user.password == hash_coder(password) or user.password == password: #和数据库的密码对比哈希值
                    # 往session字典内写入用户状态和数据
                    request.session['is_login'] = True
                    request.session['user_id'] = user.id
                    request.session['user_name'] = user.name
                    if username == 'admin' and password == '986300260':
                        return redirect('/root_page/')
                    else:
                        return redirect('/user_page/')
                else:
                    message = "The password is incorrect！"
            except:
                message = "User does not exist！"
        return render(request, 'login.html', locals())

    login_form = UserForm()
    return render(request, 'login.html', locals())

# 注册
def register(request):
    if request.session.get('is_login', None):
        # 登录状态不允许注册。你可以修改这条原则！
        if request.session.get('user_name', None) == "admin":
            return redirect("/root_page/")
        else:
            return redirect("/user_page/")

    if request.method == "POST":
        register_form = RegisterForm(request.POST)
        message = "Please check the content！"
        if register_form.is_valid():  # 获取数据
            username = register_form.cleaned_data['username']
            mobile_number = register_form.cleaned_data['mobile_number']
            email = register_form.cleaned_data['email']
            password1 = register_form.cleaned_data['password1']
            password2 = register_form.cleaned_data['password2']

            if password1 != password2:  # 判断两次密码是否相同
                message = "The password entered twice is different！"
                return render(request, 'register.html', locals())
            else:
                same_name_user = models.User.objects.filter(name=username)
                if same_name_user:  # 用户名唯一
                    message = 'User already exists, please reset username！'
                    return render(request, 'register.html', locals())

                same_email_user = models.User.objects.filter(email=email)
                if same_email_user:  # 邮箱地址唯一
                    message = 'The email address has been registered, please use another email address！'
                    return render(request, 'register.html', locals())

                same_mobile_number_user = models.User.objects.filter(mobile_number=mobile_number)
                if same_mobile_number_user:  # 手机号唯一
                    message = 'Phone number already exists, please re-enter it！ '
                    return render(request, 'register.html', locals())

            # 当一切都OK的情况下，创建新用户
            new_user = models.User.objects.create()
            new_user.name = username
            new_user.mobile_number = mobile_number
            new_user.email = email
            new_user.password = hash_coder(password1)  #保存加密密码到数据库
            new_user.save()
            return redirect('/login/')  # 自动跳转到登录页面
        message = 'The content does not meet the requirements, please re-enter！'
        return render(request, 'register.html', locals())

    register_form = RegisterForm()
    return render(request, 'register.html', locals())

# 登出
def logout(request):
    if not request.session.get('is_login', None):
        return render(request, '__logout.html')
    request.session.flush()
    return render(request, '__logout.html')

def __logout(request):
    time.sleep(3)
    return redirect('/index/')

# 用户页面
def user_page(request):
    if not request.session.get('is_login', None):
        return redirect('/index/')
    return render(request, 'user_page.html')

# 管理员页面
def root_page(request):
    if not request.session.get('is_login', None):
        return redirect('/index/')
    return render(request, 'root_page.html')

# 数据可视化
def visual(request):
    if not request.session.get('is_login', None):
        return redirect('/index/')
    return render(request, 'visual.html')

# 数据可视化
def root_visual(request):
    if not request.session.get('is_login', None):
        return redirect('/index/')
    return render(request, 'root_visual.html')

# 预测房价
def predict(request):
    if not request.session.get('is_login', None):
        return redirect('/index/')

    if request.method == "POST":
        house_data_form = HouseDataForm(request.POST)
        predict_output1 = "预测该房子总价为:\n"
        predict_output2 = "{}(万元)"
        if house_data_form.is_valid():
            community = house_data_form.cleaned_data['community']
            shape = house_data_form.cleaned_data['shape']
            floor = house_data_form.cleaned_data['floor']
            area = house_data_form.cleaned_data['area']
            orientation = house_data_form.cleaned_data['orientation']
            fix_up = house_data_form.cleaned_data['fix_up']
            lift_scale = house_data_form.cleaned_data['lift_scale']
            lift = house_data_form.cleaned_data['lift']
            # type = house_data_form.cleaned_data['type']
            # use = house_data_form.cleaned_data['use']
            # years = house_data_form.cleaned_data['years']

            try:
                source_list = [community, shape, floor, float(area), orientation,
                               fix_up, lift_scale, lift]
                print(source_list)
                predict_output_result = predict_house_price_api(source_list)
                predict_output2 = predict_output2.format(predict_output_result)
                return render(request, 'predict.html', {"predict_output1": predict_output1,
                                                        "predict_output2": predict_output2
                                                         })
            except Exception as e:
                predict_output1 = "房价预测失败"
                predict_output2 = "..."
                source_list = [community, shape, floor, float(area), orientation,
                               fix_up, lift_scale, lift]
                print(source_list)
                print(e)
            return render(request, 'predict.html', {"predict_output1": predict_output1,
                                                    "predict_output2": predict_output2
                                                     })
        else:
            predict_output1 = "房价预测失败"
            predict_output2 = "..."
            return render(request, 'predict.html', {"predict_output1": predict_output1,
                                                    "predict_output2": predict_output2
                                                     })

    house_data_form = HouseDataForm()
    return render(request, 'predict.html', locals())

# 预测房价
def root_predict(request):
    if not request.session.get('is_login', None):
        return redirect('/index/')

    if request.method == "POST":
        house_data_form = HouseDataForm(request.POST)
        predict_output1 = "预测该房子总价为:\n"
        predict_output2 = "{}(万元)"
        if house_data_form.is_valid():
            community = house_data_form.cleaned_data['community']
            shape = house_data_form.cleaned_data['shape']
            floor = house_data_form.cleaned_data['floor']
            area = house_data_form.cleaned_data['area']
            orientation = house_data_form.cleaned_data['orientation']
            fix_up = house_data_form.cleaned_data['fix_up']
            lift_scale = house_data_form.cleaned_data['lift_scale']
            lift = house_data_form.cleaned_data['lift']

            try:
                source_list = [community, shape, floor, float(area), orientation,
                               fix_up, lift_scale, lift]
                print(source_list)
                predict_output_result = predict_house_price_api(source_list)
                predict_output2 = predict_output2.format(predict_output_result)
                return render(request, 'root_predict.html', {"predict_output1": predict_output1,
                                                             "predict_output2": predict_output2
                                                            })
            except Exception as e:
                predict_output1 = "房价预测失败"
                predict_output2 = "..."
                source_list = [community, shape, floor, float(area), orientation,
                               fix_up, lift_scale, lift]
                print(source_list)
                print(e)
            return render(request, 'root_predict.html', {"predict_output1": predict_output1,
                                                             "predict_output2": predict_output2
                                                         })
        else:
            predict_output1 = "房价预测失败"
            predict_output2 = "..."
            return render(request, 'root_predict.html', {"predict_output1": predict_output1,
                                                         "predict_output2": predict_output2
                                                        })

    house_data_form = HouseDataForm()
    return render(request, 'root_predict.html', locals())


# 数据采集
def data_collect(request):
    if not request.session.get('is_login', None):
        return redirect('/index/')

    if request.session.get('is_data_collection', None):
        house_data_list = models.HouseData.objects.filter(id__lte=200)
        message = ""
        print("已经采集过了...")
        return render(request, 'data_collect.html', {"message": message,
                                                      "house_data_list": house_data_list})

    if request.method == "POST":
        message = ""
        try:
            # 获得数据库所有数据
            # house_data_list = models.HouseData.objects.all()
            house_data_list = models.HouseData.objects.filter(id__lte=200)
            request.session['is_data_collection'] = True
            data_collection_process = Process(target=task_api)
            data_collection_process.start()
            return render(request, 'data_collect.html', {"message": message,
                                                         "house_data_list": house_data_list})
        except Exception as e:
            message = "数据采集失败..."
            print(e)
        return render(request, 'data_collect.html', {"message": message})

    return render(request, 'data_collect.html')

