# Python3Predictor
1. 简介
+ 基于python3的磁定位方法，使用多个（当前为9个）三轴磁传感器，通过测量小磁铁产生的磁场，基于无迹卡尔曼滤波（UKF）来预测位置（x, y , z）和姿态（四元数q0, q1, q2, q3）
2. sensor
+ 采用ST的LIS2MDL三轴传感器，量程为±49Gs，最小分辨率为1.5mGs
3. 算法流程
+ 原始数据处理：磁传感器输出的值包含环境磁场、低磁场和噪声，因此做了清零和平滑处理
+ UKF预测目标位置和朝向
+ 3D显示坐标、姿态
