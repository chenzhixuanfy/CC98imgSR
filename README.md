在 cc98 上传图片时，无论是否选择无损上传，图片都会进行压缩，分辨率下降，变得模糊。

本项目首先下载cc98经过压缩的图片（可以实时下载新帖的图片或者下载指定id中的所有图片），然后使用超分辨率技术补充图像细节。

# 配置环境（可选）

`conda create --name cc98pic python=3.9`

`conda activate cc98pic`

`pip install -r requirements.txt`

# 图片下载
## Preparing
在项目根目录下创建`account.json`，并填入自己的 CC98 用户名和密码：
```json
{
    "username": "cc98用户名",
    "password": "cc98密码"
}
```

## 单纯下载图片
运行`CC98imgDownload.py`

输入以下命令查看帮助：
`python CC98imgDownload.py -h`

# 超分辨率图片生成
## 模型训练
1. 添加数据集，结构为
```
├─ ./CocoData/
│ ├─ train2017/
│ ├─ val2017/
```
2. 运行`create_data_lists.py`创建文件列表
3. 使用残差网络超分辨率重建。训练残差网络：`python train_srresnet.py`
4. 使用GAN网络超分辨率重建。训练 GAN 网络：`python train_srgan.py`
5. 使用`eval.py`、`test.py`评估模型

## 下载cc98图片并生成高分辨率图片
分辨率高的图片后缀为sr

模型文件路径：`./results/checkpoint_srgan.pth`
运行`CC98imgSR.py`
以下命令查看帮助：
`python CC98imgSR.py -h`