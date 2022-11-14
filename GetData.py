#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/11/3 5:04 下午
# @Author  : 李萌周
# @Email   : 983852772@qq.com
# @File    : GetData.py
# @Software: PyCharm
# @remarks : 获取数据
import math
import time

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from Model import MyModel
from 代码.Connect_DataBase import BaseSet
from 可视化 import *
database = BaseSet()
def GetData():
    sql = []
    sql.append("select * , substring_index(substring_index( date, ' ', - 1 ),':',1) time from hb_bis_himawari_AOD where date<'2022-06-30 23:00:00' order by date")
    parameter = None
    Aerosol_data = database.Get_Aerosol_train(sql,parameter)
    pd_Data = pd.DataFrame(Aerosol_data,columns=['id', 'date', 'type', 'spacing', 'speed','time'])
    pd_Data = pd_Data.astype({'time': 'int'})
    pd_Data = pd_Data.astype({'speed': 'int'})
    return pd_Data
def Data_Process(pd_Data):
    # 过滤气溶胶为0的数据
    result_data = pd_Data
    result_data = result_data[result_data["spacing"] > 0]
    # 过滤较大的值
    result_data = result_data[result_data["spacing"] < 1.5]
    # 删除掉没用的列数据
    result_data = result_data.drop("id", axis=1)
    result_data = result_data.drop("date", axis=1)
    result_data = result_data.drop("type", axis=1)
    # result_data = result_data.drop("time", axis=1)

    # 没有归一化的初始数据
    original = result_data  # 初始数据

    # 单特征归一化
    global Single_feature_scaler
    temp_data = result_data.drop("time", axis=1)
    temp_data = temp_data.drop("speed", axis=1)
    Single_feature_scaler = MinMaxScaler(feature_range=(0, 1))  # 定义缩放 默认0 1
    Single_feature_scaler.fit(temp_data)

    # 多特征归一化
    global Multi_feature_scaler
    Multi_feature_scaler = MinMaxScaler(feature_range=(0, 1))  # 定义缩放 默认0 1
    Multi_feature_scaler.fit(result_data)  # 计算最大值和最小值
    result_data = Multi_feature_scaler.transform(result_data)
    data_raw = result_data
    '''
    #数据归一化 返归一化的时候用到，因为另一个回一化输入是2维的 这个是一维的
    global one_to_one_scaler   # 这个归一化是1维的
    one_to_one_scaler = MinMaxScaler(feature_range=(0, 1))  # 定义缩放 默认0 1
    # one_to_one_scaler.fit(original.drop("time", axis=1))  # 计算最大值和最小值
    # scaler features of x according to features_range

    #数据归一化
    global scaler_x
    scaler_x = MinMaxScaler(feature_range=(0, 1))  # 定义缩放 默认0 1
    scaler_x.fit(result_data)  # 计算最大值和最小值
    # scaler features of x according to features_range
    result_data = scaler_x.transform(result_data)
    data_raw = result_data
    '''


    '''数据分组 设置回环'''
    lookback = 6
    # 转换成ndarry数组
    # data_raw = result_data.to_numpy()
    data = []
    for index in range(len(data_raw) - lookback):  # lockback为窗口大小
        data.append(data_raw[index: index + lookback])
    data = np.array(data)  # (6524, 8, 1)

    '''多特征训练预测 数据'''
    Multi_feature = []
    '''划分测试数据和训练数据'''
    test_set_size = int(np.round(0.08 * data.shape[0])) #1305
    train_set_size = data.shape[0] - (test_set_size)
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    x_test = data[train_set_size:, :-1,:]
    y_test = data[train_set_size:, -1, :]
    Multi_feature.append(x_train)
    Multi_feature.append(y_train[:,0].reshape(-1,1))   # 去掉时间维度特征 因为out_dim只有一个维度 所以标签也只能有一个维度
    Multi_feature.append(x_test)
    Multi_feature.append(y_test[:,0].reshape(-1,1))
    #一个维度的时候
    '''单特征数据 只有历史数据的特征'''
    Single_feature = []
    x_train = x_train[:,:,0]
    x_train = np.expand_dims(x_train,2)# 升维 输入特征为3维
    y_train = y_train[:,0].reshape(-1,1)
    x_test = x_test[:,:,0]
    x_test = np.expand_dims(x_test,2)# 升维 输入特征为3维
    y_test = y_test[:,0].reshape(-1,1)
    Single_feature.append(x_train)
    Single_feature.append(y_train)
    Single_feature.append(x_test)
    Single_feature.append(y_test)


    return Single_feature,Multi_feature,original


# 模型训练函数
def Model_Train(num_epochs=200, x_train=None, y_train=None, model=None, criterion=None, optimiser=None):
    # 数据转换为tensor类型
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)

    train_loss_Score_list = []  # 保存误差
    start_time = time.time()
    hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train)
        # calculate root mean squared error
        # trainScore = mean_squared_error(y_train.detach().numpy(), y_train_pred.detach().numpy())
        train_loss_Score = math.sqrt(mean_squared_error(y_train.detach().numpy(), y_train_pred.detach().numpy()))

        if (t + 1) % 100 == 0:
            # print("Epoch ", t+1, "MSE: ", loss.item())
            print("Epoch ", t + 1, "MSE: ", train_loss_Score)
            train_loss_Score_list.append(train_loss_Score)
        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))

    return model, y_train_pred, train_loss_Score_list
# 模型训练函数
def Model_Train2(num_epochs=200,x_train=None,y_train=None,model=None,criterion=None,optimiser=None,x_test=None,y_test=None):
    # 数据转换为tensor类型
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test[:16,]).type(torch.Tensor)
    y_test = torch.from_numpy(y_test[:16,]).type(torch.Tensor)
    train_loss_Score_list = [] # 保存误差
    test_loss_Score_list = []  # 保存误差
    start_time = time.time()
    hist = np.zeros(num_epochs)
    train_loss_Score=0
    # 训练模型
    # model.train()
    for t in range(num_epochs):
        model.train()
        batchsize = int(x_train.shape[0]/16)
        for i in range(batchsize):
            x = x_train[i*16:(i+1)*16]
            y_train_pred = model(x)
            y = y_train[i*16:(i+1)*16]
            loss = criterion(y_train_pred, y)
            #calculate root mean squared error
            # trainScore = mean_squared_error(y_train.detach().numpy(), y_train_pred.detach().numpy())
            # train_loss_Score = math.sqrt(mean_squared_error(y_train[i*16:(i+1)*16].detach().numpy(), y_train_pred.detach().numpy()))
            # hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            # print("i:", i, "loss", loss.item())
        # with torch.no_grad():
        # 训练数据的预测结果
        with torch.no_grad():
            model.eval()
            y_train_pred = model(x_train)
            train_loss_Score = criterion(y_train,y_train_pred)
            # train_loss_Score = math.sqrt(
            #     mean_squared_error(y_train.detach().numpy(), y_train_pred.detach().numpy()))
            if t==0:
                # print("Epoch ", t + 1, "MSE: ", train_loss_Score)
                train_loss_Score_list.append(math.sqrt(train_loss_Score.item()))
            if (t+1)%10==0:
                # print("Epoch ", t+1, "MSE: ", loss.item())
                # print("train Epoch ", t+1 , "MSE: ", train_loss_Score)
                train_loss_Score_list.append(math.sqrt(train_loss_Score.item()))

            # 验证模型    with torch.no_grad():
            model.eval()
            y_test_pred = model(x_test)
            test_loss_Score = criterion(y_test, y_test_pred)
            # test_loss_Score = math.sqrt(
            #     mean_squared_error(y_test.detach().numpy(), y_test_pred.detach().numpy()))
            if t==0:
                print("Epoch:",t+1,"---Train Loss:",math.sqrt(train_loss_Score.item()),"-----Test_Epoch:",math.sqrt(test_loss_Score.item()))
                # print("train Epoch ", t+1 , "MSE: ", train_loss_Score,"---Test_Epoch ", t + 1, "MSE: ", test_loss_Score)
                test_loss_Score_list.append(math.sqrt(test_loss_Score.item()))
            if (t+1)%10==0:
                print("Epoch:", t + 1, "---Train Loss:", math.sqrt(train_loss_Score.item()), "-----Test_Epoch:", math.sqrt(test_loss_Score.item()))
                # print("Epoch ", t+1, "MSE: ", loss.item())
                # print("train Epoch ", t+1 , "MSE: ", train_loss_Score,"---Test_Epoch ", t+1 , "MSE: ", test_loss_Score)
                test_loss_Score_list.append(math.sqrt(test_loss_Score.item()))


    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))
    torch.save(model,"../Result/model.pth")
    return model,y_train_pred,train_loss_Score_list,test_loss_Score_list



# 使用测试数据验证模型
def Model_validation(model,x_test,y_test):

    y_test_pred = model(torch.from_numpy(x_test).type(torch.Tensor))


    # 如果不归一化 则用下边代码 我来试试 键盘还好用不，好像很容易跑啊，这该怎么版呢
    # y_train_pred = y_train_pred.detach().numpy()
    # y_train = y_train.detach().numpy()
    # y_test_pred = y_test_pred.detach().numpy()
    # y_test = y_test.detach().numpy()

    # 还原归一化
    # output=2 所需代码
    y_test = Single_feature_scaler.inverse_transform(y_test)
    # y_test_pred = scaler_x.inverse_transform(y_test_pred.detach().numpy())
    # output=1 所需代码
    # y_test = one_to_one_scaler.inverse_transform(y_test)
    # 还原归一化
    y_test_pred = Single_feature_scaler.inverse_transform(y_test_pred.detach().numpy())
    MSE = math.sqrt(mean_squared_error(y_test,y_test_pred))
    MAE = math.sqrt(mean_absolute_error(y_test,y_test_pred))
    R_square = r2_score(y_test,y_test_pred)
    return y_test_pred,MSE,MAE,R_square



if __name__ == "__main__":
    # 链接数据库查询数据
    data = GetData()
    # 数据预处理  # original用于可视化的部分
    # x_train,y_train,x_test,y_test,original = Data_Process(data)
    Single_feature, Multi_feature, original = Data_Process(data)


    # 准备训练
    model_parameters={
        "lr":0.0001,
        "epochs":400
    }
    '''单特征训练预测部分'''
    Single_Model = MyModel(input_dim=1,hidden_dim=32,num_layers=2,output_dim=1)
    # 损失函数
    Single_criterion = torch.nn.MSELoss()  # 均方损失函数
    # optimiser = torch.optim.Adam(model.parameters(), lr=parameter.get("lr"))
    # “Adam,一种基于低阶矩的自适应估计的随机目标函数一阶梯度优化算法
    Single_optimiser = torch.optim.Adam(Single_Model.parameters(), lr=model_parameters.get("lr"))

    # 开始训练
    Single_x_train = Single_feature[0]
    Single_y_train = Single_feature[1]
    Single_x_test = Single_feature[2]
    Single_y_test = Single_feature[3]
    Single_Model.train()

    #单特征训练结果
    # 返归一化  Single_y_train=lable  Single_y_train_pred=predict_result  Single_train_loss:train_loss
    Single_model,Single_y_train_pred,Single_train_loss,Single_test_loss = Model_Train2(model_parameters.get('epochs'),Single_x_train,Single_y_train,
                                                   Single_Model,Single_criterion,Single_optimiser,Single_x_test,Single_y_test)
    Single_y_train = Single_feature_scaler.inverse_transform(Single_y_train)
    Single_y_train_pred = Single_feature_scaler.inverse_transform(Single_y_train_pred.detach().numpy())
    # plot_multi_comparision(Single_y_train, Single_y_train_pred,name="单特征值训练可视化")
    # 验证单特征模型
    Single_Model.eval()
    Single_x_test = Single_feature[2]
    Single_y_test = Single_feature[3]
    Single_y_test_pred,Single_MSE,Single_MAE,Single_R_square= Model_validation(Single_Model, Single_x_test, Single_y_test)

    '''多特征训练预测部分'''
    Multi_Model = MyModel(input_dim=3,hidden_dim=32,num_layers=2,output_dim=1)
    # 损失函数
    Multi_criterion = torch.nn.MSELoss()  # 均方差损失函数
    # optimiser = torch.optim.Adam(model.parameters(), lr=parameter.get("lr"))
    # “Adam,一种基于低阶矩的自适应估计的随机目标函数一阶梯度优化算法
    Multi_optimiser = torch.optim.Adam(Multi_Model.parameters(), lr=model_parameters.get("lr"))

    # 开始训练
    Multi_x_train = Multi_feature[0]
    Multi_y_train = Multi_feature[1]
    Multi_x_test = Multi_feature[2]
    Multi_y_test = Multi_feature[3]
    Multi_Model.train()
    # 多特征训练结果 predict:Multi_y_train_pred,train_loss:Multi_train_loss  lable:Single_y_train
    Multi_Model, Multi_y_train_pred,Multi_train_loss,Multi_test_loss = Model_Train2(model_parameters.get('epochs'), Multi_x_train, Multi_y_train,
                                                    Multi_Model,Multi_criterion,Multi_optimiser,Multi_x_test,Multi_y_test)
    # Multi_y_train = Multi_feature_scaler.inverse_transform(Multi_y_train)
    Multi_y_train_pred = Single_feature_scaler.inverse_transform(Multi_y_train_pred.detach().numpy())
    # plot_multi_comparision(Single_y_train, Single_y_train_pred,Multi_y_train_pred)

    # 验证多特征模型
    Multi_Model.eval()
    Multi_x_test = Multi_feature[2]
    Multi_y_test = Multi_feature[3]
    Multi_y_test_pred,Multi_MSE,Multi_MAE,Multi_R_square= Model_validation(Multi_Model, Multi_x_test,Multi_y_test)
    print("Single MSE:",Single_MSE)
    print("Single MAE:", Single_MAE)
    print("Multi R-square:", Single_R_square)
    print("Multi MSE:", Multi_MSE)
    print("Multi MAE:", Multi_MAE)
    print("Multi R-square:", Multi_R_square)




    ''''可视化部分'''
    # 单特征 训练结果可视化 包含训练损失函数
    Single_train_result=[]
    Single_train_result.append(Single_y_train)
    Single_train_result.append(Single_y_train_pred)
    Single_plot_train(Single_train_result,Single_train_loss,Single_test_loss,"单特征训练结果可视化")
    #单特征 验证部分可视化
    Single_test_result = []
    Single_test_result.append(Single_y_test)
    Single_test_result.append(Single_y_test_pred)
    Single_plot_test(Single_test_result,"单特征验证结果可视化")


    # 多特征训练可视化部分
    # 包含多特征训练结果和
    # plot_train(Single_y_train, Single_y_train_pred,Multi_y_train_pred,name="单特征值训练可视化")
    train_data = [] #训练数据部分
    train_data.append(Single_y_train)#标签
    train_data.append(Single_y_train_pred)  # 单特征预测结果
    train_data.append(Multi_y_train_pred)  # 多特征预测结果
    Multi_plot_train(train_data, name="多特征训练结果可视化")

    # 多特征验证结果可视化 和损失函数
    test_data = [] #验证数据部分
    test_data.append(Single_y_test)  # 标签
    test_data.append(Single_y_test_pred)  # 单特征预测结果
    test_data.append(Multi_y_test_pred)  # 多特征预测结果
    Multi_test_result_loss = []
    Multi_test_result_loss.append(Single_test_loss)
    Multi_test_result_loss.append(Multi_test_loss)
    Multi_plot_test(test_data,Multi_test_result_loss,"多特征验证可视化")

    Multi_plot_test_comparision(test_data,Multi_test_result_loss,"单、多特征验证对比可视化")


    #可视化显示损失函数
    # plot_loss(Single_train_loss,Multi_train_loss)
    #可视化显示预测结果 对比显示
    # plot_multi_comparision(Single_y_test, Single_y_test_pred, Multi_y_test_pred)




