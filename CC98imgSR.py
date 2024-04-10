import requests
import re
import os
import sys
import time
import datetime
import threading
import random
import json
from PIL import Image
from io import BytesIO
import argparse
import torch
from utils import *
from models import SRResNet, Generator
import time
from PIL import Image

# 一些典型帖子id：4142494、3609055

class CC98PIC:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
        }
        self.post_data = {
                            "client_id": "9a1fd200-8687-44b1-4c20-08d50a96e5cd",
                            "client_secret": "8b53f727-08e2-4509-8857-e34bf92b27f2",
                            "grant_type": "password",
                            "scope": "cc98-api openid offline_access"
                        }# POST表单，这里的这些都是常值
        self.latest_id = None# 最新的帖子id
        self.last_id = None# 上一次读过的帖子id
        self.URL_title = "https://api.cc98.org/topic/{id}" # 因为URL_content获得的title变成了null，所以需要从这里获取
        self.URL_content = "https://api.cc98.org/Topic/{id}/post?from=0&size=1" # from是开始的楼数，size是返回总楼数（最多好像是20）
        self.URL_auth = "https://openid.cc98.org/connect/token"# 获取Authorization的地址
        self.URL_newTopics = "https://api.cc98.org/topic/new?from=0&size=1"# 获取最新的帖子
        self.timer = None# 别忘了定义
        self.period = 60 # 执行周期，单位：秒
        self.img_root_path = 'img/'
    
    # 获得Authorization，不知道client_id和client_secret多久更新一次
    def get_auth(self):
        self.write_log("Getting authorization...")
        response_auth = requests.post(self.URL_auth, data=self.post_data, headers=self.headers)
        if(response_auth):
            settings = response_auth.json()
            self.headers["Authorization"] = settings["token_type"] + " " + settings["access_token"]
            self.write_log("Authorization is ready. ")
        else:
            self.write_log("Fail to get! ")
            time.sleep(5)
            self.get_auth()
    
    def get_method(self, URL, headers):
        response = requests.get(URL, headers = headers)
        # self.write_log("status code = " + str(response.status_code))
        if(response.status_code != 200):# 授权过期
            self.get_auth()
        else:
            return response
        
    def get_latest_id(self):
        response = self.get_method(self.URL_newTopics, self.headers)
        data = response.json()
        self.latest_id = data[0]["id"]
        self.write_log("latest id: " + str(self.latest_id))
    
    def error_handle(self):
        except_type, except_value, except_traceback = sys.exc_info()
        except_file = os.path.split(except_traceback.tb_frame.f_code.co_filename)[1]
        exc_dict = {
            "报错类型": except_type,
            "报错信息": except_value,
            "报错文件": except_file,
            "报错行数": except_traceback.tb_lineno,
        }
        self.write_log(str(exc_dict))

    # 写入日志并打印
    def write_log(self, text):
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
        print(text)

    # 获取帖子标题
    def get_title(self, id):
        try:
            response = self.get_method(self.URL_title.format(id=id), headers = self.headers)
            data = response.json()
            title = data["title"]
        except Exception:
            self.error_handle()
            title = None
        
        return title
    
    # 下载webp格式的图片，并存储为jpeg格式
    def img_download(self, pic_url, save_path):
        # 使用正则表达式提取日期和文件名部分
        # \d{4} 匹配4位数字年份, \d{1,2} 匹配1位或者2位数字月份和日期, \w+ 匹配一个或多个字母数字字符组成的文件名
        pattern = r'-(\d{4}-\d{1,2}-\d{1,2})-(\w+)\.'
        # 使用 re.search 查找匹配的子字符串
        matches = re.search(pattern, pic_url.replace('/', '-')) # 为了适应不同的url格式，将/全部替换为-
        
        # 如果找到匹配项，则提取
        if matches:
            date, filename = matches.groups()
            save_name = date + '-' + filename + '.jpg'
            try:
                response = self.get_method(pic_url, self.headers)

                if 'webp' in pic_url: # webp格式的图片
                # 使用Pillow库打开图片
                    webp_image = Image.open(BytesIO(response.content))             
                    # 转换图片格式为JPEG
                    rgb_im = webp_image.convert('RGB')             
                    # 保存图片为JPEG格式
                    rgb_im.save(save_path + save_name, 'JPEG')
                else: # jpg格式的图片
                    # 打开文件写入图片数据
                    with open(save_path + save_name, 'wb') as f:
                        f.write(response.content)
                
                # 超分辨率处理
                self.img_sr(save_path + save_name)

            except Exception:
                self.error_handle()
    
    # 超分辨率
    def img_sr(self, img_path):
        img = Image.open(img_path, mode='r')
        img = img.convert('RGB')

        # 图像预处理
        lr_img = convert_image(img, source='pil', target='imagenet-norm')
        lr_img.unsqueeze_(0)
        
        # 转移数据至设备
        lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
        
        # 模型推理
        with torch.no_grad():
            sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
            sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
            sr_img.save(img_path.replace('.jpg', '') + "-sr" + '.jpg')

    # 爬一条帖子
    def crawl_1post(self, id):
        response = self.get_method(self.URL_content.format(id=id), self.headers)
        try:
            data = response.json()

            # self.write_log(data[0]["title"] + " " + "https://www.cc98.org/topic/" + str(data[0]["topicId"])) # 现在的帖子 title 都变成null了
            self.write_log("https://www.cc98.org/topic/" + str(data[0]["topicId"])) # 现在的帖子 title 都变成null了

            # 不同格式的帖子代码。典型帖子id：5855172、3870053、3609055、3690188
            pattern = re.compile(r'\[img\](.*?)\[/img\]|\[upload=jpg\](.*?)\[/upload\]|\[upload=jpg,0\](.*?)\[/upload\]|\[upload=jpg,1\](.*?)\[/upload\]')

            matched_URLs = pattern.findall(data[0]["content"]) # [(url11, url12, url13), (url21, url22, url23), ...] 的形式
            
            pic_URLs = [item for t in matched_URLs for item in t if item] # 提取出url存进列表

            # 帖子包含图片时才会生成新的文件夹
            if len(pic_URLs):
                title = self.get_title(id)
                img_path = self.img_root_path + title + '-' + str(id) + '/'
                self.create_dirs(img_path)

                for i in range(len(pic_URLs)):
                    pic_url = pic_URLs[i]

                    self.img_download(pic_url, img_path)

        except Exception:
            self.error_handle()
            data = None

        return data

    # 定时爬新帖
    def crawling(self):
        self.write_log(str(datetime.datetime.now()))
        try:
            self.get_latest_id()
        except Exception:
            self.error_handle()

            return

        for id in range(self.last_id, self.latest_id):
            data = self.crawl_1post(id+1)

            # if data == None:
            #     continue

            time.sleep(5)
        self.last_id = self.latest_id

        self.write_log("\n")

    # 定时执行（实际是隔随机时间执行）
    def func_timer(self):

        random_time = random.uniform(-self.period * 0.3, self.period * 0.3)# 小数的秒数，更不容易被发现
        
        self.crawling()

        # 定时器构造函数主要有2个参数，第一个参数为时间，第二个参数为函数名
        self.timer = threading.Timer(self.period + random_time, self.func_timer)   # 每过一段随机时间（单位：秒）调用一次函数

        self.timer.start()    #启用定时器
    
    # 生成不存在的路径
    def create_dirs(self, directory_name):
        # 获取当前工作目录
        current_directory = os.getcwd()
        # 指定新目录名为 img
        img_directory = os.path.join(current_directory, directory_name)
        # 检查目录是否存在
        if not os.path.exists(img_directory):
            # 如果不存在，创建目录
            os.makedirs(img_directory)

    # 初始化并开始爬虫   
    def crawler_start(self, mode = 0, id = 3609055):
        # 新建log文件
        with open("log.txt", "w"):
            pass
        # 读取文件数据
        with open("account.json", 'r', encoding="utf-8") as f:
            account = json.load(f)
        
        self.post_data["username"] = account["username"]
        self.post_data["password"] = account["password"]

        self.write_log("Crawler starts...\n")

        self.create_dirs('img')
        
        self.get_auth()

        if mode == 0:
            # 只爬一个帖子
            self.crawl_1post(id=id)
        else:
            # 定时爬新帖
            self.get_latest_id()
            self.last_id = self.latest_id - 5# 一开始先爬5个帖子
            self.timer = threading.Timer(1, self.func_timer)
            self.timer.start()

def parser_args():
    parser = argparse.ArgumentParser(description='抓取cc98图片并超分辨率还原细节')

    # 通过在参数名前加'--'，设置为可选参数，如果未输入，则使用default默认值（若未设置default，则会默认赋值None）
    # 通过'-'将可选参数设置引用名，可以缩短参数名
    # metavar 在通过-h显示 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.。这里通过设置为空一律不显示。
    parser.add_argument('-m', '--mode', type=int, choices=[0, 1], default=0, metavar='',
                        help='脚本执行模式：mode=0，下载一个特定帖子的图片；mode=1，持续下载新帖图片。默认mode=0。')
    parser.add_argument('-i', '--id', type=int, default=3609055, metavar='',
                        help='当mode=0时，指定抓取的帖子id。')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parser_args()

    # 模型参数
    large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
    small_kernel_size = 3   # 中间层卷积的核大小
    n_channels = 64         # 中间层通道数
    n_blocks = 16           # 残差模块数量
    scaling_factor = 4      # 放大比例
    # device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 预训练模型
    srgan_checkpoint = "./results/checkpoint_srgan.pth"
    # srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srgan_checkpoint, map_location='cpu')
    generator = Generator(large_kernel_size=large_kernel_size,
                          small_kernel_size=small_kernel_size,
                          n_channels=n_channels,
                          n_blocks=n_blocks,
                          scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()
    model = generator

    cc98_pic = CC98PIC()
    cc98_pic.crawler_start(mode=args.mode, id=args.id)