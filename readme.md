## The background of the project
Our school encourages undergraduate research and can sign up for the corresponding program. The teacher we chose happened to be teaching graduate students about computer vision, so she also encouraged us to do a computer vision project. So there was this project, and my team and I finished it in '24 and used it to participate in some competitions.


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

