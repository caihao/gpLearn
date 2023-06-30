# GPLearn

## 介绍

GPLearn是我们针对切伦科夫望远镜所独立编写的程序包，基于python语言和pytorch模型，我们将不同的功能单独打包成相对应的模块，并允许用户根据需求自定义模块参数。
通过GPLearn进行的一系列数据模拟与分析的相关结果详见，读者不仅可以利用我们提供的数据还原文章中展示的结果，同时也可以用于实际探测器的数据分析，我们希望这一结果可以被广泛用于天文物理领域。

## 软件架构

为了您可以正常使用GPLearn，请确保您在当前环境下安装了以下依赖：
python  3.8.5
pytorch  1.7.0
torchvision  0.8.0
numpy  1.22.2
pandas  1.2.4

## 安装教程

1.  在您指定的位置上运行如下克隆命令：<br><center>**git clone https://gitee.com/chengaoyan/gplearn.git**</center>
2.  如果您是首次下载使用，请首先运行位于程序住目录下的**init.py**文件，该脚本会检查程序主路径下的**/data**目录及其子目录是否完整（如不完整则会创建补全）
3.  位于程序主目录下的**main.py**文件是作业脚本，您可以根据您的不同需求在其中设定程序的执行流程，一些参数的调整可能需要您改动**settings.json**文件，具体方法将在以下部分详细阐述。此外，**main_quick_start.py**提供了适用于背景抑制工作的简单演示程序，您可以直接运行
4.  我们更推荐您使用_nohup_命令在超算服务器上托管运行，为此，我们编写了适配的日志功能与剩余时间预测功能，您的所有输出结果（包括所有中间结果）都将被自动保存

## 使用说明

1.  xxxx
2.  xxxx
3.  xxxx

## 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


## 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
