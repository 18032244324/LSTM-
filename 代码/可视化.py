#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/11/7 5:36 下午
# @Author  : 李萌周
# @Email   : 983852772@qq.com
# @File    : 可视化.py
# @Software: PyCharm
# @remarks : 无

# 可视训练的化损失函数

from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
#单特征 训练结果可视化
def Single_plot_train(Single_train_result,Single_train_loss,Single_test_loss,name="单特征训练结果可视化"):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_ylim(-0.2, 2)
    ax1.set_title("单特征训练结果")
    ax1.set_ylabel('Aerosol value')
    ax1.set_xlabel('Hour')
    ax1.plot(Single_train_result[0][0:300], linestyle='-', linewidth=1, label='真实值',color="black")
    ax1.plot(Single_train_result[1][:300], linestyle='-.', linewidth=1, label='预测值',color="black")
    # plt.legend(loc='lower right', fontsize=40)
    plt.legend(fontsize=8)
    ax2 = fig.add_subplot(2,1,2)
    # ax2.set_ylim(-0.2, 2)
    ax2.set_title("训练损失VS验证损失")
    ax2.plot(Single_train_loss, linestyle='-', linewidth=1, label='训练损失',color='black')
    ax2.plot(Single_test_loss, linestyle='-.', linewidth=1, label='验证损失', color='black')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch：10/次')
    plt.tight_layout()
    plt.legend(fontsize=10)
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']SimHei
    # plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.savefig('../Result/'+name+'.jpg', bbox_inches='tight', dpi=450)
    plt.show()
    plt.close()
#单特征 验证结果可视化
def Single_plot_test(Single_test_result,name="单特征验证结果可视化"):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_ylim(-0.2, 2)
    ax1.set_title("单特征测试结果")
    ax1.set_ylabel('Aerosol value')
    ax1.set_xlabel('Hour')
    plt.plot(Single_test_result[0], linestyle='-', linewidth=1, label='真实值',color='black')
    plt.plot(Single_test_result[1], linestyle='-.', linewidth=1, label='预测值', color='black')
    plt.tight_layout()
    plt.legend()
    plt.savefig('../Result/'+name+'.jpg', bbox_inches='tight', dpi=450)
    plt.show()
    plt.close()


#多特征 训练结果可视化
def Multi_plot_train(Multi_train_result,name="多特征训练结果可视化"):
    fig = plt.figure()
    # 单特训练结果
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax1.set_ylim(-0.2, 2)
    # # ax1.set_title("Single feature train result")
    # ax1.set_title("单特征训练结果")
    # ax1.set_ylabel('Aerosol value')
    # ax1.set_xlabel('Hour')
    # ax1.plot(Multi_train_result[0], linestyle='-', linewidth=1, label='真实值', color='black')
    # ax1.plot(Multi_train_result[1], linestyle=':', linewidth=1, label='预测值', color='black')
    # plt.legend(fontsize=7)
    ax2 = fig.add_subplot(2, 1, 1)
    ax2.set_ylim(-0.2, 2)
    ax2.set_title("单特征训练结果top100")
    ax2.set_ylabel('Aerosol value')
    ax2.set_xlabel('Hour')
    ax2.plot(Multi_train_result[0][0:100], linestyle='-', linewidth=1, label='真实值', color='black')
    ax2.plot(Multi_train_result[1][0:100], linestyle='-.', linewidth=1, label='预测值', color='black')
    plt.legend(fontsize=8)
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax3.set_ylim(-0.2, 2)
    # ax3.set_title("多特征训练结果")
    # ax3.set_ylabel('Aerosol value')
    # ax3.set_xlabel('Hour')
    # ax3.plot(Multi_train_result[0], linestyle='-', linewidth=1, label='真实值', color='black')
    # ax3.plot(Multi_train_result[2], linestyle=':', linewidth=1, label='预测值', color='black')
    # plt.legend(fontsize=7)
    ax4 = fig.add_subplot(2, 1, 2)
    ax4.set_ylim(-0.2, 2)
    ax4.set_title("多特征训练结果top100")
    ax4.set_ylabel('Aerosol value')
    ax4.set_xlabel('Hour')
    ax4.plot(Multi_train_result[0][0:100], linestyle='-', linewidth=1, label='真实值', color='black')
    ax4.plot(Multi_train_result[2][0:100], linestyle='-.', linewidth=1, label='预测值', color='black')
    plt.legend(fontsize=8)  #现实label
    plt.tight_layout() # 调整布局
    plt.savefig('../Result/'+name+'.jpg', bbox_inches='tight', dpi=450)
    plt.show()
    plt.close()

#多特征 验证结果可视化
def Multi_plot_test(Multi_test_result,Multi_train_loss,name="多特征验证结果可视化"):
    fig = plt.figure()
    # 单特征验证结果
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_ylim(-0.2, 2)
    ax1.set_title("单特征测试结果")
    ax1.set_ylabel('Aerosol value')
    ax1.set_xlabel('Hour')
    ax1.plot(Multi_test_result[0], linestyle='-', linewidth=1, label='真实值',color='black')
    ax1.plot(Multi_test_result[1], linestyle='-.', linewidth=1, label='预测值',color='black')
    plt.legend()
    # 多特征验证结果
    ax3 = fig.add_subplot(2,1,2)
    ax3.set_ylim(-0.2, 2)
    ax3.set_title("多特征测试结果")
    ax3.set_ylabel('Aerosol value')
    # ax3.set_xlabel('Epoch')
    ax3.set_xlabel('Hour')
    ax3.plot(Multi_test_result[0], linestyle='-', linewidth=1, label='真实值',color='black')
    ax3.plot(Multi_test_result[2], linestyle='-.', linewidth=1, label='预测值', color='black')
    plt.legend()
    '''
    # 单特征损失函数
    ax2 = fig.add_subplot(2,2,2)
    # ax2.set_ylim(-0.2, 2)
    ax2.set_title("Single train loss")
    ax2.set_ylabel('Loss')
    ax2.plot(Multi_train_loss[0], linestyle='-', linewidth=1, label='loss',color='green')
    plt.legend()
    # 多特征损失函数
    ax4 = fig.add_subplot(2,2,4)
    ax4.set_title("Multi train loss")
    ax4.set_ylabel('Loss')
    ax4.plot(Multi_train_loss[1], linestyle='-', linewidth=1, label='loss', color='green')
    plt.legend()
    '''
    plt.tight_layout()  # 调整布局
    plt.savefig('../Result/'+name+'.jpg', bbox_inches='tight', dpi=450)
    plt.show()
    plt.close()

def Multi_plot_test_comparision(Multi_test_result,Multi_train_loss,name="多特征损失函数对比可视化"):
    # 验证结果对比
    fig = plt.figure()
    # 单特征验证结果
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_ylim(-0.2, 2)
    ax1.set_title("单、多特征预测结果对比")
    ax1.set_ylabel('Aerosol value')
    ax1.set_xlabel('Hour')
    ax1.plot(Multi_test_result[0], linestyle='-', linewidth=0.7, label='真实值', color='black')
    ax1.plot(Multi_test_result[1], linestyle=':', linewidth=0.7, label='单特征预测', color='black')
    ax1.plot(Multi_test_result[2], linestyle='-.', linewidth=0.7, label='多特征预测', color='black')
    plt.legend()
    # 损失函数对比
    ax2 = fig.add_subplot(2,1,2)
    # ax2.set_ylim(-0.2, 2)
    ax2.set_title("损失函数对比")
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch：10/次')
    ax2.plot(Multi_train_loss[0], linestyle=':', linewidth=1, label='单特征损失',color='black')
    ax2.plot(Multi_train_loss[1], linestyle='-.', linewidth=1, label='多特征损失', color='black')
    plt.legend()
    plt.tight_layout()  # 调整布局
    plt.savefig('../Result/'+name+'.jpg', bbox_inches='tight', dpi=450)
    plt.show()
    plt.close()
'''
def plot_loss(Single_train_loss,Multi_train_loss):
    plt.plot(Single_train_loss, linestyle='-', linewidth=1, label='Single_loss')
    plt.plot(Multi_train_loss, linestyle='-', linewidth=1, label='Multi__loss')
    plt.legend()
    plt.savefig('./损失函数结果.jpg', bbox_inches='tight', dpi=450)
    plt.show()
    plt.close()

# 训练结果可视化
def plot_train_comparision(train_data,test_data,name=""):
    fig = plt.figure()
    #训练部分可视化
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_ylim(-0.2, 2)
    ax1.set_title("Single_feature_pred")
    # plt.title("title")
    ax1.plot(train_data[0], linestyle='-', linewidth=1, label='Real',color='red')
    ax1.plot(train_data[1], linestyle='-', linewidth=1, label='Predict', color='blue')
    #训练部分 可视化前200数据
    ax2 = fig.add_subplot(2,1,3)
    ax2.set_ylim(-0.2, 2)
    ax2.set_title("Multi_feature_pred")
    ax2.plot(train_data[0], linestyle='-', linewidth=1, label='Real', color='red')
    ax2.plot(train_data[2], linestyle='-', linewidth=1, label='Predict', color='blue')
    plt.legend()
    plt.savefig('./'+name+'jpg', bbox_inches='tight', dpi=450)
    plt.show()
    plt.close()
# 结果预测结果可视化
def plot_multi_comparision(real=None,Single_pred=[],Multi_pred=[],name=""):  # pred 是cnn的预测结果

    # fig = plt.figure(figsize=(7, 7))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_ylim(-0.25, 2.5)
    # my_x_ticks = np.arange(1, 750, 50)
    # plt.xticks(my_x_ticks)
    # ax.set_xlim(0,1000)
    # fig.set_ylim(-1,3)

    plt.plot(real, linestyle='-', linewidth=1, label='Real')
    if len(Multi_pred)>0:
        plt.plot(Single_pred, linestyle='-', linewidth=1, label='Single_Predict')
        plt.plot(Multi_pred, linestyle='-', linewidth=1, label='Multi_Predict')
    else:
        plt.plot(Single_pred, linestyle='-', linewidth=1, label='Single_Predict')
    # plt.legend(fontsize=17)
    plt.legend()
    # plt.yticks([])
    # plt.xticks([])
    plt.tight_layout()
    # plt.savefig(png_fn, dpi=300)
    # plt.cla()
    plt.savefig('./'+name+'jpg', bbox_inches='tight', dpi=450)
    plt.show()
    plt.close()
'''