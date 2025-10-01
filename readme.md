# 项目说明 / Project Description

[🇨🇳 中文说明](#中文说明) | [🇬🇧 English Description](#english-description)

---

## 中文说明

### 项目的背景
我们的学校鼓励本科生科研，可以报名相关项目。我们选择的导师刚好在给研究生讲授计算机视觉课程，因此也鼓励我们做一个计算机视觉的课题。于是我们就开展了这个项目。我们在 2024 年完成了它，并利用该项目参加了2024中国计算机设计大赛并获得国赛三等奖。  

### 项目的主要功能
该项目主要实现了 **人类面部表情识别**，我们使用了一些现有的高效模型来完成这一任务。在此基础上，我们设计了一个界面，并在获得表情分析结果后进行了进一步的计算与推理。例如：当疲劳表情在一定次数内的比例高于某个阈值时，系统会发出提示。  
该软件的设计目标是 **提升计算机用户的开发效率**，并帮助他们进行情绪健康的分析与管理。  

### 如何使用本程序

**运行环境**  
   - windows amd64系统
   - mysql >= 5.0


#### 直接下载程序
1. 点击进入release界面，选择对应版本的程序。

2. 保存`asset\setting.txt`文件，并修改其中的数据库账户名和密码。

3. 将下载好的程序和`setting.txt`放到同一文件夹下即可运行。


#### 自己训练模型并使用
1. **安装依赖**
   - 确保已安装 requirements 文件中列出的依赖
2. **配置数据库**  
   - 打开 `asset` 文件夹中的 `setting` 文件  
   - 将用户名和密码改为你自己的 MySQL 账户   

3. **模型训练**  
   - 在 `model_CNN.py` 文件中修改数据集路径为你自己的数据集（如果有的话），或者直接使用本项目提供的数据集。  
   - 在 `train` 文件夹中运行 `model_CNN.py` 文件，并确保将模型保存到项目根目录下的 `model` 文件夹，命名为 `model_cnn.pkl`  

4. **运行用户界面**  
   - 直接运行 `ui` 文件夹中的 `backup_ui.py` 文件即可 
   - 你也可以打包为exe等形式的程序，直接运行 


### 待完善

- [ ] 发布第一版exe程序
- [ ] 脱离setting文件，允许用户直接与程序交互更新配置

---
## English Description

## The background of the project
Our school encourages undergraduate research and can sign up for the corresponding program. The teacher we chose happened to be teaching graduate students about computer vision, so she also encouraged us to do a computer vision project. So there was this project, and my team and I finished it in '24. This project was utilized to participate in the 2024 China Computer Design Competition and won the third prize in the national competition.

## The main function of the project
This project mainly realizes the recognition of human facial expressions, and we use some existing efficient models to accomplish this task. On this basis, we designed an interface, and completed the further calculation and reasoning after obtaining the expression analysis, for example, when the proportion of tired expressions appears higher than a certain threshold within a certain number of times, a prompt will be issued. This software is designed to improve the development efficiency of computer users and help them complete the analysis and management of emotional health.

## How to use the program
**Operating environment**
   - mysql >= 5.0
   - windows amd64

### download the program directly
1. Click to enter the release interface and select the corresponding version of the program.

2. Save the `asset\setting.txt` file and modify the database account name and password in it.

3. Just put the downloaded program and `setting.txt` in the same folder and you can run it.



### Train the model yourself and use it
1. **Install the required dependencies**
And make sure that the dependencies described in the requiements file are installed

2. **Configure the database Settings:**
Open the `setting.txt` file in the `asset` folder and change the username and password to your mysql account

3. **Model training**
- Modify the dataset path in the `model_CNN.py` file to your dataset (if you have one)，Or you can directly use the dataset provided by this project.
- Run the `model_CNN.py` file in the `train` folder and make sure to save the model to the `model` folder in the root directory and name it `model_cnn.pkl`

4. **Run the user interface**
- Just run the `backup_ui.py` file directly in the `ui` folder
- You can also package it into an exe program and run it directly


## To be improved

- [ ] Released the first version of the exe program
- [ ] Detached from the setting file, allowing users to directly interact with the program to update the configuration

## About us
- The person in charge of the project: Li Yang, 2718014695@qq.com  
 He is the first to propose and improve the project idea, and is also the main participant in the implementation of algorithms and data collation.    


- The project member, Xu Fuquan,  1469767416@qq.com
 was mainly involved in the algorithm implementation and test analysis, and he read the source code of a model in depth and understood its principle.  
   

- The project member, Yu Qingrong,  YuQingrong@qq.com  
  The database design and interface design of this project were largely completed by her, and she is a main participant in the implementation of the algorithms.  


- The project member, Li Jingyi,  1121480941@qq.com
    She is the main participant in the data collection and collation of this project, and is the main participant in the algorithm implementation and database design.


- The project member, Luan Xiaorui,  2801611461@qq.com
   He is the main participant in the application development, testing and analysis of this project.

