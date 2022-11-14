import pymysql
import time
class BaseSet():
    # 连接数据库基础参数
    def __init__(self):
        # 连接数据库
        self.conn = pymysql.connect(host='127.0.0.1'  # 连接名称，默认127.0.0.1   192.168.32.102
                                    , user='root'  ## 用户名
                                    , passwd='rootroot'  # 密码2008-CFwork.HBCF._
                                    , port=3306  # 端口，默认为3306
                                    , db='test'  # 数据库名称ry-vue
                                    , charset='utf8'  # 字符编码
                                    )
        self.cur = self.conn.cursor()  # 生成游标对象

    def GetCamIDInfo(self,sql,parameter):
        try:
            self.cur.execute(sql[0],parameter)  # 执行插入的sql语句
            row = self.cur.execute(sql[1],parameter)  # 执行插入的sql语句
            if row>0:
                self.cur.execute(sql[2], parameter)
            # self.conn.close()
            # self.cur.close()
            return self.cur.fetchall()
        except:
            self.conn.commit()  # 提交到数据库执

    # 这个在用 耗时40s  获取日溶胶和对应的pm 用来做回归预测   现在数据库表结果更新了  不需要很久了 10.8号
    def Get_Aerosol_PM(self,sql,parameter):
        try:
            a = self.cur.execute(sql[0],parameter)  # 执行插入的sql语句 设置创建临时表
            # b = self.cur.execute(sql[1])    # 创建临时表  耗时40S
            # c = self.cur.execute(sql[2],parameter)  #查询临时表所需的内容
            # i=1+parameter[1]
            # while c==0:
            #     if i>=88:
            #         i=i-88
            #     c = self.cur.execute(sql[2], parameter[0].get("imgid")[i])
            #     i=i+1

            self.conn.commit()
            return self.cur.fetchall()
        except:
            # self.conn.commit()  # 提交到数据库执
            print("获取训练数据出错")

    # 获取气溶胶数据，然后使用网络进行预测
    def Get_Aerosol(self,sql,parameter):
        self.cur.execute(sql[0], parameter)
        self.conn.commit()
        return self.cur.fetchall()

    # 保存预测的pm10浓度值
    def save_prd_pm10(self,sql,parameter):
        # 执行插入语句
        try:
            r = self.cur.executemany(sql[0],parameter)
            self.conn.commit()
            # print(r)
        except:
            print("保存预测的pm10浓度值出错")

    # 保存预测的pm25浓度值
    def save_prd_pm25(self,sql,parameter):
        # 执行插入语句
        r = self.cur.execute(sql[0],parameter)
        return self.cur.fetchall()

    # 查询真是的pm值和预测的 然后可视化现实误差
    def view_pm_and_predict(self,sql,parameter):
        # 执行查询语句 看看temp的联合表还在不在
        a = self.cur.execute(sql[0], parameter)
        print(a)

    '''论文代码从这里开始'''
    # 保存气溶胶数据的部分
    def Save_Satellite_Aerosol(self,sql, parameter):
        self.cur.execute(sql[0], parameter)
        self.conn.commit()

    # 获取数据用于训练模型
    def Get_Aerosol_train(self,sql, parameter):
        self.cur.execute(sql[0], parameter)
        self.conn.commit()
        return self.cur.fetchall()
    def save_weather(self,sql,parameter):
        # for i in range((len(parameter))):
        #     a = parameter[i]
        self.cur.execute(sql[0],parameter)
        self.conn.commit()
    def close_database(self):
        self.cur.close()
        self.conn.close()
