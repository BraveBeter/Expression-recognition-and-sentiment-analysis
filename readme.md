# 项目说明 / Project Description

[🇨🇳 中文说明](#中文说明) | [🇬🇧 English Description](#english-description)

---

## 中文说明

### 项目的背景
我们的学校鼓励本科生科研，可以报名相关项目。我们选择的导师刚好在给研究生讲授计算机视觉课程，因此也鼓励我们做一个计算机视觉的课题。于是我们就开展了这个项目。我和团队成员在 2024 年完成了它，并利用该项目参加了2024中国计算机设计大赛并获得国赛三等奖。  

### 项目的主要功能
该项目主要实现了 **人类面部表情识别**，我们使用了一些现有的高效模型来完成这一任务。在此基础上，我们设计了一个界面，并在获得表情分析结果后进行了进一步的计算与推理。例如：当疲劳表情在一定次数内的比例高于某个阈值时，系统会发出提示。  
该软件的设计目标是 **提升计算机用户的开发效率**，并帮助他们进行情绪健康的分析与管理。  

### 如何使用本程序
1. **运行环境**  
   - mysql >= 5.0  
   - 确保已安装 requirements 文件中列出的依赖  

2. **配置数据库**  
   - 打开 `asset` 文件夹中的 `setting` 文件  
   - 将用户名和密码改为你自己的 MySQL 账号  

3. **模型训练**  
   - 在 `model_CNN` 文件中修改数据集路径为你自己的数据集（如果有的话）  
   - 在 `train` 文件夹中运行 `model_CNN` 文件，并确保将模型保存到项目根目录下的 `model` 文件夹，命名为 `model_cnn.pkl`  
  > 或者你可以跳过这一步，直接联系我，我会将我们已经训练好的模型文件发送给你（`.pkl` 格式）  

4. **运行用户界面**  
   - 直接运行 `ui` 文件夹中的 `backup_ui` 文件即可  
---
## English Description

## The background of the project
Our school encourages undergraduate research and can sign up for the corresponding program. The teacher we chose happened to be teaching graduate students about computer vision, so she also encouraged us to do a computer vision project. So there was this project, and my team and I finished it in '24. This project was utilized to participate in the 2024 China Computer Design Competition and won the third prize in the national competition.

## The main function of the project
This project mainly realizes the recognition of human facial expressions, and we use some existing efficient models to accomplish this task. On this basis, we designed an interface, and completed the further calculation and reasoning after obtaining the expression analysis, for example, when the proportion of tired expressions appears higher than a certain threshold within a certain number of times, a prompt will be issued. This software is designed to improve the development efficiency of computer users and help them complete the analysis and management of emotional health.

## How to use the program

1. **Operating environment:**
mysql >= 5.0
And make sure that the dependencies described in the requiements file are installed

2. **Configure the database Settings:**
Open the "setting" file in the "asset" folder and change the username and password to your mysql account

3. **Model training**
- Modify the dataset path in the model_CNN file to your dataset (if you have one).
- Run the model_CNN file in the train folder and make sure to save the model to the model folder in the root directory and name it model_cnn.pkl
> Or you can skip this step and contact me directly. I will send you the model file we have trained directly. The file type is pkl

4. **Run the user interface**
Just run the backup_ui file directly in the ui folder


## About us
- The person in charge of the project: Li yang, 2718014695@qq.com  
 He is the first to propose and improve the project idea, and is also the main participant in the implementation of algorithms and data collation.    


- The project member, Xu Fuquan,  1469767416@qq.com
 was mainly involved in the algorithm implementation and test analysis, and he read the source code of a model in depth and understood its principle.  
   

- The project member, Yu Qingrong,  YuQingrong@qq.com  
  The database design and interface design of this project were largely completed by her, and she is a main participant in the implementation of the algorithms.  


- The project member, Li Jingyi,  1121480941@qq.com
    She is the main participant in the data collection and collation of this project, and is the main participant in the algorithm implementation and database design.


- The project member, Luan Xiaorui,  Lxr@qq.com
   He is the main participant in the application development, testing and analysis of this project.

