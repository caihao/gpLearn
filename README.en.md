# GPLearn

#### Description

GPLearn is a package we wrote independently for the Cherenkov telescope. Based on the python language and the pytorch model, we packaged the different functions into separate modules and allowed the user to customize the module parameters according to their needs.
The results of a series of data simulations and analyses performed through GPLearn are detailed in the article, and the reader can use the data we provide not only to restore the results shown in the article, but also to analyze the data from the actual detector, which we hope will be widely used in the field of astrophysics.

#### Software Architecture

In order for you to use GPLearn properly, please make sure you have the following dependencies installed in your current environment:
python 3.8.5
pytorch 1.7.0
torchvision 0.8.0
numpy 1.22.2
pandas 1.2.4

#### Installation

1.  Run the following clone command on the location you specified:<br><center>**git clone https://gitee.com/chengaoyan/gplearn.git**</center>
2.  If this is your first download, please first run the file **init.py** located in the program's residence directory. This script will check if the directory **/data** and its subdirectories under the program's main path are complete (if not, a completion will be created)
3.  The file **main.py** located in the program's main directory is a job script where you can set up the execution process of the program according to your different needs. Some parameter adjustments may require you to change the file **settings.json**, as explained in detail in the following sections. In addition, file **main_quick_start.py** provides a simple demo program for background suppression work, which you can run directly
4.  We prefer that you use the _nohup_ command to run hosted on the supercomputing server, for which we have written an adapted logging function with remaining time prediction, and all your output (including all intermediate results) will be saved automatically

#### Instructions

1.  xxxx
2.  xxxx
3.  xxxx

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
