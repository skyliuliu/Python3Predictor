# Python3Predictor

## 1. 简介
　　基于python3的磁定位方法，使用多个（当前为9个）三轴磁传感器，通过测量小磁铁产生的磁场，基于无迹卡尔曼滤波（UKF）来预测位置（x,
y , z）和姿态（四元数q0, q1, q2, q3）

## 2. sensor
　　采用ST的LIS2MDL三轴传感器，量程为±49Gs，最小分辨率为1.5mGs

## 3. 算法流程

+ 原始数据处理：磁传感器输出的值包含环境磁场、低磁场和噪声，因此做了清零和平滑处理
+ UKF预测目标位置和朝向
+ 3D显示坐标、姿态

## 4. 文件组成

##### 4.1 readData.py
+ 读取PCB板的通信数据
+ 绘图函数plotMag（原始B值和平滑后的）和plotB(原始B值和预测B值)
+ 磁偶极矩的公式定义
+ 配置参数的定义，包括：sensor个数、磁矩大小、sensor之间的距离等
##### 4.2 magPredictor.py
　　实现预测胶囊功能的主文件，调用UKF库和其它文件
##### 4.3 dataViewer.py
　　3D显示磁矩的位置和姿态
##### 4.4 momentPredictor.py
　　预测磁矩的大小，用于磁矩未知的对象，获得磁矩值
##### 4.5 trajectoryView.py
　　3D显示磁矩的运动轨迹
##### 4.6 sesnorVar.xlsx
　　传感器噪声随着外加磁场的测试结果
##### 4.7 bg.json
　　保存的背景磁场
##### 4.8 requirements.txt
　　依赖的第三方库名称
##### 4.9 EPMPredictor.py
　　专用于预测外部磁体

## 5. 使用步骤
（1）在readData.py配置好相应的参数<br>
（2）运行momentPredictor.py获取精确的磁矩值，然后修改readData.py中的MOMENT<br>
（3）运行magPredictor.py进行预测，可选择不同的进程，以启动观测程序<br>
`注：运行时出现"go on?Calibrate ok!"的提示后可将胶囊放置好，然后按下回车键`