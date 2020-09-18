from django import forms  #表单
from captcha.fields import CaptchaField

# 注册用户表单
class RegisterForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=128,
                               widget=forms.TextInput(attrs={'class': 'form-control'}))
    mobile_number = forms.CharField(label="手机号", max_length=11,
                                    widget=forms.TextInput(attrs={'class': 'form-control'}))
    email = forms.EmailField(label="邮箱地址", widget=forms.EmailInput(attrs={'class': 'form-control'}))
    password1 = forms.CharField(label="密码", max_length=256,
                                widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    password2 = forms.CharField(label="确认密码", max_length=256,
                                widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    captcha = CaptchaField(label="验证码")

# 用户信息表单
class UserForm(forms.Form):
    username = forms.CharField(label="用户名", max_length=128, widget=forms.TextInput(attrs={'class': 'form-control'}))
    password = forms.CharField(label="密码", max_length=256, widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    captcha = CaptchaField(label='验证码')

class HouseDataForm(forms.Form):
    community = forms.CharField(label="小区名称", max_length=128,
                                widget=forms.TextInput(attrs={'class': 'form-control'}))
    shape = forms.CharField(label="房屋户型", max_length=128,
                            widget=forms.TextInput(attrs={'class': 'form-control'}))
    floor = forms.CharField(label="所在楼层", max_length=128,
                            widget=forms.TextInput(attrs={'class': 'form-control'}))
    area = forms.CharField(label="建筑面积",
                           widget=forms.TextInput(attrs={'class': 'form-control'}))
    orientation = forms.CharField(label="房屋朝向", max_length=128,
                                  widget=forms.TextInput(attrs={'class': 'form-control'}))
    fix_up = forms.CharField(label="装修情况", max_length=128,
                             widget=forms.TextInput(attrs={'class': 'form-control'}))
    lift = forms.CharField(label="配备电梯", max_length=128,
                           widget=forms.TextInput(attrs={'class': 'form-control'}))
    type = forms.CharField(label="交易权属", max_length=128,
                           widget=forms.TextInput(attrs={'class': 'form-control'}))
    use = forms.CharField(label="房屋用途", max_length=128,
                          widget=forms.TextInput(attrs={'class': 'form-control'}))
    years = forms.CharField(label="房屋年限", max_length=128,
                            widget=forms.TextInput(attrs={'class': 'form-control'}))
