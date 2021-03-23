import sys
import ui5 as myui
import mul_lstm1 as t3
import mul_lstm_train1 as t4
import automl_ui as at
import parameter_list as para
import auto_test2 as auto
import pred
import precode2 as p
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import lstm_nn_test2 as t1

import lstm_nn_train2 as t2
import pandas as pd
import login as log
import dataframe as mydf

import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QBasicTimer
import Recmendation

class MyRecmendation(QWidget, Recmendation.Ui_Form):
    def __init__(self, parent=None):
        super(MyRecmendation, self).__init__(parent)
        self.setupUi(self)
        #self.setStyleSheet("#Form{border-image:url(image/bg1.jpg);}")
        self.pushButton_back.clicked.connect(self.click_back)
        self.pushButton_recommend.clicked.connect(self.recommend)
        self.count_1 = 0
        self.count_2 = 0
        self.count_3 = 0
        self.dic_close = {"maotai":0.0754,"tonghuashun":0.0732,"wanke":0.0333,"zhenye":0.0322,"yuanye":0.0396,"baoan":0.0478,
                          "anda":0.0345,"jinxing":0.09540}
        self.dic_open = {"maotai":0.0754,"tonghuashun":0.0632,"wanke":0.0295,"zhenye":0.0431,"yuanye":0.0224,"baoan":0.0321,
                         "anda":0.0268,"jinxing":0.09882}


    def click_back(self):
        self.setVisible(False)
        ui.main_win.setVisible(True)


    def Strategy(self):
        self.stock_name = [' ']
        self.predClose = []
        self.predOpen = []
        self.realClose = []
        self.acc_close = []
        self.acc_open = []
        if self.checkBox_maotai.isChecked():
            self.stock_name.append("mt")
            data_openPrice = np.array(pd.read_csv("dataset_recommend/dataset_openPrice/maotai.csv"))
            data_closePrice = np.array(pd.read_csv("dataset_recommend/dataset/maotai.csv"))
            x = len(data_closePrice)-1
            y = len(data_closePrice[0])-1
            self.realClose.append(data_closePrice[x,y])
            temppred_close = p.fun(TIME_STEPS=20,BATCH_SIZE=50,CELL_SIZE=60, LSTM_LAYER=3,NN_LAYER=1,restorePath="model_maotaiclose/",
                                   data=data_closePrice,NN_CELLSIZE=40,LSTM_OUTPUT=1)
            self.predClose.append(temppred_close)
            temppred_open = p.fun(TIME_STEPS=20,BATCH_SIZE=50,CELL_SIZE=60, LSTM_LAYER=3,NN_LAYER=1,restorePath="model_maotaiopen/",
                                   data=data_openPrice,NN_CELLSIZE=40,LSTM_OUTPUT=1)
            self.predOpen.append(temppred_open)
            self.acc_close.append(self.dic_close["maotai"])
            self.acc_open.append(self.dic_open["maotai"])
        if self.checkBox_tonghuashun.isChecked():
            self.stock_name.append("ths")
            data_openPrice = np.array(pd.read_csv("dataset_recommend/dataset_openPrice/tonghuashun.csv"))
            data_closePrice = np.array(pd.read_csv("dataset_recommend/dataset/tonghuashun1.csv"))

            x = len(data_closePrice)-1
            y = len(data_closePrice[0])-1
            self.realClose.append(data_closePrice[x,y])
            temppred_close = p.fun(TIME_STEPS=20,BATCH_SIZE=50,CELL_SIZE=40, LSTM_LAYER=2,NN_LAYER=2,restorePath="model_tonghuashunclose/",
                                   data=data_closePrice,NN_CELLSIZE=15,LSTM_OUTPUT=1)
            self.predClose.append(temppred_close)
            temppred_open = p.fun(TIME_STEPS=20,BATCH_SIZE=50,CELL_SIZE=40, LSTM_LAYER=2,NN_LAYER=2,restorePath="model_tonghuashunopen/",
                                   data=data_openPrice,NN_CELLSIZE=15,LSTM_OUTPUT=1)
            self.predOpen.append(temppred_open)
            self.acc_close.append(self.dic_close["tonghuashun"])
            self.acc_open.append(self.dic_open["tonghuashun"])
        if self.checkBox_wanke.isChecked():
            self.stock_name.append("wk")
            data_openPrice = np.array(pd.read_csv("dataset_recommend/dataset_openPrice/wanke.csv"))
            data_closePrice = np.array(pd.read_csv("dataset_recommend/dataset/wanke.csv"))
            x = len(data_closePrice) - 1
            y = len(data_closePrice[0])-1
            self.realClose.append(data_closePrice[x, y])
            temppred_close = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=2, NN_LAYER=1,
                                  restorePath="model_wankeclose/",
                                  data=data_closePrice, NN_CELLSIZE=30, LSTM_OUTPUT=1)
            self.predClose.append(temppred_close)
            temppred_open = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=50, LSTM_LAYER=3, NN_LAYER=2,
                                  restorePath="model_wankeopen/",
                                  data=data_openPrice, NN_CELLSIZE=20, LSTM_OUTPUT=1)
            self.predOpen.append(temppred_open)
            self.acc_close.append(self.dic_close["wanke"])
            self.acc_open.append(self.dic_open["wanke"])
        if self.checkBox_zhenye.isChecked():
            self.stock_name.append("zy")
            data_openPrice = np.array(pd.read_csv("dataset_recommend/dataset_openPrice/深振业A.csv"))
            data_closePrice = np.array(pd.read_csv("dataset_recommend/dataset/深振业A.csv"))
            x = len(data_closePrice) - 1
            y = len(data_closePrice[0])-1
            self.realClose.append(data_closePrice[x, y])
            temppred_close = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=3, NN_LAYER=1,
                                  restorePath="model_zhenyeclose/",
                                  data=data_closePrice, NN_CELLSIZE=30, LSTM_OUTPUT=1)
            self.predClose.append(temppred_close)
            temppred_open = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=3, NN_LAYER=1,
                                  restorePath="model_zhenyeopen/",
                                  data=data_openPrice, NN_CELLSIZE=30, LSTM_OUTPUT=1)
            self.predOpen.append(temppred_open)
            self.acc_close.append(self.dic_close["zhenye"])
            self.acc_open.append(self.dic_open["zhenye"])
        if self.checkBox_jinxing.isChecked():
            self.stock_name.append("jx")
            data_openPrice = np.array(pd.read_csv("dataset_recommend/dataset_openPrice/深锦兴A.csv"))
            data_closePrice = np.array(pd.read_csv("dataset_recommend/dataset/深锦兴A.csv"))
            x = len(data_closePrice) - 1
            y = len(data_closePrice[0])-1
            self.realClose.append(data_closePrice[x, y])
            temppred_close = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=3, NN_LAYER=2,
                                  restorePath="model_jingxinclose/",
                                  data=data_closePrice, NN_CELLSIZE=20, LSTM_OUTPUT=1)
            self.predClose.append(temppred_close)
            temppred_open = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=3, NN_LAYER=1,
                                  restorePath="model_jingxinopen/",
                                  data=data_openPrice, NN_CELLSIZE=30, LSTM_OUTPUT=1)
            self.predOpen.append(temppred_open)
            self.acc_close.append(self.dic_close["jinxing"])
            self.acc_open.append(self.dic_open["jinxing"])
        if self.checkBox_yuanye.isChecked():
            self.stock_name.append("yy")
            data_openPrice = np.array(pd.read_csv("dataset_recommend/dataset_openPrice/深原野A.csv"))
            data_closePrice = np.array(pd.read_csv("dataset_recommend/dataset/深原野A.csv"))
            x = len(data_closePrice) - 1
            y = len(data_closePrice[0])-1
            self.realClose.append(data_closePrice[x, y])
            temppred_close = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=3, NN_LAYER=1,
                                  restorePath="model_yuanyeclose/",
                                  data=data_closePrice, NN_CELLSIZE=30, LSTM_OUTPUT=1)
            self.predClose.append(temppred_close)
            temppred_open = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=3, NN_LAYER=1,
                                  restorePath="model_yuanyeopen/",
                                  data=data_openPrice, NN_CELLSIZE=30, LSTM_OUTPUT=10)
            self.predOpen.append(temppred_open)
            self.acc_close.append(self.dic_close["yuanye"])
            self.acc_open.append(self.dic_open["yuanye"])
        if self.checkBox_baoan.isChecked():
            self.stock_name.append("ba")
            data_openPrice = np.array(pd.read_csv("dataset_recommend/dataset_openPrice/深宝安A.csv"))
            data_closePrice = np.array(pd.read_csv("dataset_recommend/dataset/深宝安A.csv"))
            x = len(data_closePrice) - 1
            y = len(data_closePrice[0])-1
            self.realClose.append(data_closePrice[x, y])
            temppred_close = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=3, NN_LAYER=1,
                                  restorePath="model_baoanclose/",
                                  data=data_closePrice, NN_CELLSIZE=30, LSTM_OUTPUT=1)
            self.predClose.append(temppred_close)
            temppred_open = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=3, NN_LAYER=1,
                                  restorePath="model_baoanopen/",
                                  data=data_openPrice, NN_CELLSIZE=30, LSTM_OUTPUT=1)
            self.predOpen.append(temppred_open)
            self.acc_close.append(self.dic_close["baoan"])
            self.acc_open.append(self.dic_open["baoan"])
        if self.checkBox_anda.isChecked():
            self.stock_name.append("ad")
            data_openPrice = np.array(pd.read_csv("dataset_recommend/dataset_openPrice/深安达A.csv"))
            data_closePrice = np.array(pd.read_csv("dataset_recommend/dataset/深安达A.csv"))
            x = len(data_closePrice) - 1
            y = len(data_closePrice[0])-1
            self.realClose.append(data_closePrice[x, y])
            temppred_close = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=3, NN_LAYER=1,
                                  restorePath="model_andaclose/",
                                  data=data_closePrice, NN_CELLSIZE=30, LSTM_OUTPUT=1)
            self.predClose.append(temppred_close)
            temppred_open = p.fun(TIME_STEPS=20, BATCH_SIZE=50, CELL_SIZE=40, LSTM_LAYER=3, NN_LAYER=1,
                                  restorePath="model_andaopen/",
                                  data=data_openPrice, NN_CELLSIZE=30, LSTM_OUTPUT=1)
            self.predOpen.append(temppred_open)
            self.acc_close.append(self.dic_close["anda"])
            self.acc_open.append(self.dic_open["anda"])

    def recommend(self):
        self.Strategy()
        self.f_overnight = MyFigure()
        self.f_overnight.rec_overnight(closeprice=self.realClose, openprice=self.predOpen, stockname=self.stock_name)
        if self.count_1 == 0:
            self.gridlayout_1 = QGridLayout(self.groupBox_overnight)  # 继承容器groupBox
            self.gridlayout_1.addWidget(self.f_overnight, 0, 1)
        else:
            self.gridlayout_1.addWidget(self.f_overnight, 0, 1)
        self.count_1 = self.count_1 + 1

        self.f_today = MyFigure()
        self.f_today.rec_today(predclose=self.predClose, predopen=self.predOpen, stockname=self.stock_name)
        if self.count_2 == 0:
            self.gridlayout_2 = QGridLayout(self.groupBox_today)  # 继承容器groupBox
            self.gridlayout_2.addWidget(self.f_today, 0, 1)
        else:
            self.gridlayout_2.addWidget(self.f_today, 0, 1)
        self.count_2 = self.count_2 + 1

        self.f_acc = MyFigure()
        self.f_acc.rec_acc(acc_close=self.acc_close, acc_open=self.acc_open, stockname=self.stock_name)
        if self.count_3 == 0:
            self.gridlayout_3 = QGridLayout(self.groupBox_acc)  # 继承容器groupBox
            self.gridlayout_3.addWidget(self.f_acc, 0, 1)
        else:
            self.gridlayout_3.addWidget(self.f_acc, 0, 1)
        self.count_3 = self.count_3 + 1

        self.textBrowser.append("本次推荐选择的股票有：")
        self.textBrowser.append(str(self.stock_name))
        self.textBrowser.append("所选各股票收益：")
        for i in range(len(self.stock_name)-1):
            self.textBrowser.append("----------------")
            self.textBrowser.append(
                self.stock_name[i+1]+str("预测的开盘价为：")+"     "+str(self.predOpen[i])
            +"     "+str("预测的收盘价为：")+str(self.predClose[i])
            )
            self.textBrowser.append(
                str("隔夜交易的收益为：")+str(self.predClose[i]-self.realClose[i]) + "    " +
                str("当天交易的收益为：") + str(self.predClose[i]-self.predOpen[i])
                                )
        max_value_night = 0
        max_value_today = 0
        for i in range(len(self.stock_name)-1):
            if (self.predClose[i]-self.realClose[i])>(self.predClose[max_value_night]-self.realClose[max_value_night]):
                max_value_night = i
            if(self.predClose[i] - self.predOpen[i]) > (self.predClose[max_value_today] - self.predOpen[max_value_today]):
                max_value_today = i
        self.textBrowser.append("~~~~~~~~~~~~~~~~~~~~~")
        self.textBrowser.append("隔夜交易收益最高为："+self.stock_name[max_value_night+1]+"   "+
                                "收益为:"+str(self.predClose[max_value_night]-self.realClose[max_value_night]))
        self.textBrowser.append("当天交易收益最高为：" + self.stock_name[max_value_today + 1] + "   " +
                                "收益为:" + str(self.predClose[max_value_today] - self.predOpen[max_value_today]))
        self.textBrowser.append("*******finish********")






class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        super(MyFigure,self).__init__(self.fig)


    def plotpred(self, lr, batch_size, timesteps, lstm_cell, lstm_layer,nn_layer, is_test, restoringpath,
                 data,nn_cellsize , lstm_output):
        self.axes1 = self.fig.add_subplot(111)
        try:
            prediction, real, self.acc1, self.acc2, self.acc3 = t1.fun(TIME_STEPS = timesteps,BATCH_SIZE = batch_size,CELL_SIZE = lstm_cell,LSTM_LAYER = lstm_layer,
        LR = lr,NN_LAYER=nn_layer,is_test = is_test,
        restorePath = restoringpath,data = data, NN_CELLSIZE=nn_cellsize, LSTM_OUTPUT=lstm_output)
            self.axes1.plot(list(range(len(prediction))), prediction, color='black')
            self.axes1.plot(list(range(len(real))), real, color='r')

        except:
            QMessageBox.critical(self, '错误', '请重新输入!', QMessageBox.Ok)



    def plotcost(self, itera, lr, batch_size, timesteps, lstm_cell, lstm_layer, nn_layer, is_train, lstm_keep_prob, nn_keep_prob, savingpath, restoringpath,
                 data, nn_cellsize , lstm_output):
        self.axes2 = self.fig.add_subplot(111)
        try:
            cost = t2.fun(TIME_STEPS = timesteps,BATCH_SIZE = batch_size,CELL_SIZE = lstm_cell,LSTM_LAYER = lstm_layer,
        LR = lr,N_ITER = itera,KEEP_PROB_LSTM = lstm_keep_prob,KEEP_PROB_NN = nn_keep_prob,NN_LAYER=nn_layer,is_train = is_train,
        savingPath = savingpath,data = data, NN_CELLSIZE=nn_cellsize, LSTM_OUTPUT=lstm_output)
            self.axes2.plot(list(range(len(cost))), cost, color='y')
        except:
            QMessageBox.critical(self,'错误','请重新输入!',QMessageBox.Ok)



    def plotpred2(self, itera, lr, batch_size, timesteps, lstm_cell, lstm_layer, is_train, is_test, lstm_keep_prob, savingpath, restoringpath,
                 data):
        self.axes3 = self.fig.add_subplot(111)
        try:
            prediction, real, self.acc11, self.acc22, self.acc33 = t3.fun(TIME_STEPS = timesteps,BATCH_SIZE = batch_size,CELL_SIZE = lstm_cell,LSTM_LAYER = lstm_layer,
        LR = lr,is_test = is_test,restorePath = restoringpath,data = data)
            self.axes3.plot(list(range(len(prediction))), prediction, color='black')
            self.axes3.plot(list(range(len(real))), real, color='r')
        except:
            QMessageBox.critical(self, '错误', '请重新输入!', QMessageBox.Ok)

    def plotcost2(self, itera, lr, batch_size, timesteps, lstm_cell, lstm_layer, is_train, is_test, lstm_keep_prob, savingpath, restoringpath,
                 data):
        self.axes4 = self.fig.add_subplot(111)
        try:
            cost = t4.fun(TIME_STEPS = timesteps,BATCH_SIZE = batch_size,CELL_SIZE = lstm_cell,LSTM_LAYER = lstm_layer,
        LR = lr,N_ITER = itera,KEEP_PROB_LSTM = lstm_keep_prob,is_train = is_train,
        savingPath = savingpath,data = data)
            self.axes4.plot(list(range(len(cost))), cost, color='y')
        except:
            QMessageBox.critical(self,'错误','请重新输入!',QMessageBox.Ok)

    # def plotauto(self, itera, lr, batch_size, timesteps, lstm_cell, lstm_layer,nn_layer, lstm_keep_prob, nn_keep_prob, savingpath, restoringpath,
    #              data,lstm_output,nn_cellsize):
    #     self.axes0 = self.fig.add_subplot(111)
    #     try:
    #         prediction, real, self.acc1, self.acc2, self.acc3 = t_auto.fun(TIME_STEPS = timesteps,BATCH_SIZE = batch_size,CELL_SIZE = lstm_cell,LSTM_LAYER = lstm_layer,
    #     LR = lr,N_ITER = itera,KEEP_PROB_LSTM = lstm_keep_prob,KEEP_PROB_NN = nn_keep_prob,NN_LAYER=nn_layer,
    #     savingPath = savingpath,restorePath = restoringpath,data = data,LSTM_OUTPUT=lstm_output,NN_CELLSIZE=nn_cellsize)
    #         self.axes0.plot(list(range(len(prediction))), prediction, color='black')
    #         self.axes0.plot(list(range(len(real))), real, color='r')
    #     except:
    #         QMessageBox.critical(self, '错误', '请重新输入!', QMessageBox.Ok)

    def plot_data_1(self, data):
        self.axes5 = self.fig.add_subplot(111)
        close_price = pd.DataFrame(data, columns=['closePrice'])
        self.axes5.plot(list(range(len(close_price))), close_price, color='y')

    def plot_data_2(self, data):
        self.axes6 = self.fig.add_subplot(111)
        dealmount = pd.DataFrame(data, columns=['dealAmount'])
        self.axes6.plot(list(range(len(dealmount))), dealmount, color='green')

    def plot_data_3(self, data):
        self.axes7 = self.fig.add_subplot(111)
        marketValue = pd.DataFrame(data, columns=['marketValue'])
        self.axes7.plot(list(range(len(marketValue))), marketValue, color='red')

    def rec_overnight(self, closeprice, openprice, stockname):
        self.axes8 = self.fig.add_subplot(111)
        self.axes8.set_xlim(0, len(stockname))
        self.axes8.set_xticks(np.arange(0, len(stockname)))
        self.axes8.set_xticklabels(stockname)
        value = np.array(openprice) - np.array(closeprice)
        for i in range(len(value)):
            self.axes8.bar(i + 1, value[i], color="r")
        #self.axes8.bar(list(range(len(value))), value, color='red')
        self.axes8.set_xticklabels(stockname)
        self.axes8.plot(list(1+np.array(range(len(value)))),value,color="g")

    def rec_today(self, predclose, predopen, stockname):
        self.axes9 = self.fig.add_subplot(111)
        self.axes9.set_xlim(0, len(stockname))
        self.axes9.set_xticks(np.arange(0, len(stockname)))
        self.axes9.set_xticklabels(stockname)
        value = np.array(predclose) - np.array(predopen)
        for i in range(len(value)):
            self.axes9.bar(i+1,value[i],color="black")
        #self.axes9.bar(range(len(value)), value, color='black')
        self.axes9.set_xticklabels(stockname)
        self.axes9.plot(list(1 + np.array(range(len(value)))), value, color="g")

    def rec_acc(self,acc_close,acc_open,stockname):
        self.axes10 = self.fig.add_subplot(111)
        self.axes10.set_xlim(0,len(stockname))
        self.axes10.set_xticks(np.arange(0,len(stockname)))
        self.axes10.set_xticklabels(stockname)
        self.axes10.bar(list(1-0.35/2 + np.array(range(len(acc_close)))), acc_close, 0.35, color="c", label="收盘", alpha=0.5)
        self.axes10.bar(list(1+0.35/2 + np.array(range(len(acc_close)))), acc_open, 0.35, color="b",
                        label="开盘", alpha=0.5)



class   pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class Mydataframe(QMainWindow,mydf.Ui_Form):
    def __init__(self,parent=None):
        super(Mydataframe, self).__init__(parent)
        self.setupUi(self)
        self.comboBox_data.currentIndexChanged.connect(self.click_combodata)
        self.pushButton_back.clicked.connect(self.click_back)

        self.count1 = 0
        self.count2 = 0
        self.count3 = 0

    def click_combodata(self):
        self.f_1 = MyFigure()
        self.f_2 = MyFigure()
        self.f_3 = MyFigure()
        if self.comboBox_data.currentText() == "同花顺":
            data = pd.read_csv("dataset_origin/tonghuashun.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "茅台":
            data = pd.read_csv("dataset_origin/maotai.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "万科":
            data = pd.read_csv("dataset_origin/深万科A.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "振业":
            data = pd.read_csv("dataset_origin/深振业A.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "原野":
            data = pd.read_csv("dataset_origin/深原野A.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "锦兴":
            data = pd.read_csv("dataset_origin/深锦兴A.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "金田":
            data = pd.read_csv("dataset_origin/深金田A.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "发展":
            data = pd.read_csv("dataset_origin/深发展A.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "达声":
            data = pd.read_csv("dataset_origin/深达声A.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "宝安":
            data = pd.read_csv("dataset_origin/深宝安A.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "安达":
            data = pd.read_csv("dataset_origin/深安达A.csv",encoding = 'gbk')
        elif self.comboBox_data.currentText() == "恒邦股份":
            data = pd.read_csv("dataset_origin/hengbang.csv",encoding = 'gbk')
        self.f_1.plot_data_1(data)
        if self.count1 == 0:
            self.gridlayout1 = QGridLayout(self.groupBox_1)  # 继承容器groupBox
            self.gridlayout1.addWidget(self.f_1, 0, 1)
        else:
            self.gridlayout1.addWidget(self.f_1, 0, 1)
        self.count1 = self.count1 + 1

        self.f_2.plot_data_2(data)
        if self.count2 == 0:
            self.gridlayout2 = QGridLayout(self.groupBox_2)  # 继承容器groupBox
            self.gridlayout2.addWidget(self.f_2, 0, 1)
        else:
            self.gridlayout2.addWidget(self.f_2, 0, 1)
        self.count2 = self.count2 + 1

        self.f_3.plot_data_3(data)
        if self.count3 == 0:
            self.gridlayout3 = QGridLayout(self.groupBox_3)  # 继承容器groupBox
            self.gridlayout3.addWidget(self.f_3, 0, 1)
        else:
            self.gridlayout3.addWidget(self.f_3, 0, 1)
        self.count3 = self.count3 + 1

        self.model = pandasModel(data)
        self.tableView.setModel(self.model)
    def click_back(self):
        ui.main_win.setVisible(True)
        self.setVisible(False)



class MyWinodw2(QMainWindow,at.Ui_AutoForm):
    def __init__(self,parent=None):
        super(MyWinodw2,self).__init__(parent)
        self.setupUi(self)
        self.setStyleSheet("#AutoForm{border-image:url(image/bg1.jpg);}")
        #self.textBrowser.setStyleSheet("#textBrowser{border-image:url(image/bg5.jpeg);}")
        self.pushButton_hands.clicked.connect(self.click_back_to_hands)
        self.pushButton_update.clicked.connect(self.click_update)
        self.pushButton_3.clicked.connect(self.click_begin_auto)

        self.textBrowser_showbatch.setText(str(para.config.batch_size))
        self.textBrowser_showtimesteps.setText(str(para.config.time_steps))
        self.textBrowser_showcellsize.setText(str(para.config.cell_size))
        self.textBrowser_showtlr.setText(str(para.config.lr))
        self.textBrowser_showiter.setText(str(para.config.n_iter))
        self.textBrowser_showkpnn.setText(str(para.config.keep_prob_nn))
        self.textBrowser_keep_prob_lstm.setText(str(para.config.keep_prob_lstm))
        self.textBrowser_show_nn_layer.setText(str(para.config.nn_layer))
        self.textBrowser_showlstm_layer.setText(str(para.config.lstm_layer))
        self.textBrowser_showlstm_output.setText(str(para.config.lstm_output))
        self.textBrowser_show_nn_cellsize.setText(str(para.config.nn_cellsize))
        self.lineEdit_save.setText("auto_test/model.ckpt")
        self.lineEdit_restore.setText("auto_test/")

        self.g_batch= QButtonGroup(self)
        self.g_batch.addButton(self.radioButton_batch1)
        self.g_batch.addButton(self.radioButton_batch2)
        self.g_timestep = QButtonGroup(self)
        self.g_timestep.addButton(self.radioButton_timesteps1)
        self.g_timestep.addButton(self.radioButton_timesteps2)
        self.g_cellsize = QButtonGroup(self)
        self.g_cellsize.addButton(self.radioButton_cellsize1)
        self.g_cellsize.addButton(self.radioButton_cellsize2)
        self.g_lr= QButtonGroup(self)
        self.g_lr.addButton(self.radioButton_lr1)
        self.g_lr.addButton(self.radioButton_lr2)
        self.g_iter = QButtonGroup(self)
        self.g_iter.addButton(self.radioButton_iter1)
        self.g_iter.addButton(self.radioButton_liter2)
        self.g_kpnn= QButtonGroup(self)
        self.g_kpnn.addButton(self.radioButton_kpnn1)
        self.g_kpnn.addButton(self.radioButton_kpnn2)
        self.g_kplstm = QButtonGroup(self)
        self.g_kplstm.addButton(self.radioButton_kplstm1)
        self.g_kplstm.addButton(self.radioButton_lkplstm2)
        self.g_nnlayer = QButtonGroup(self)
        self.g_nnlayer.addButton(self.radioButton_nn_layer1)
        self.g_nnlayer.addButton(self.radioButton_nn_layer2)
        self.g_lstmlayer = QButtonGroup(self)
        self.g_lstmlayer.addButton(self.radioButton_lstm_layer1)
        self.g_lstmlayer.addButton(self.radioButton_lstm_layer2)
        self.g_lstm_output = QButtonGroup(self)
        self.g_lstm_output.addButton(self.radioButton_lstm_output1)
        self.g_lstm_output.addButton(self.radioButton_lstm_output2)
        self.g_nn_cellsize = QButtonGroup(self)
        self.g_nn_cellsize.addButton(self.radioButton_nn_cellsize1)
        self.g_nn_cellsize.addButton(self.radioButton_nn_cellsize2)

        self.count_pred = 0
    def click_update(self):
        if self.lineEdit_batch.text()!='':
            if self.radioButton_batch1.isChecked() == True and (int(self.lineEdit_batch.text()) not in para.config.batch_size):
                para.config.batch_size.append(int(self.lineEdit_batch.text()))
            if self.radioButton_batch2.isChecked() == True:
                para.config.batch_size.remove(int(self.lineEdit_batch.text()))
            self.textBrowser_showbatch.setText(str(para.config.batch_size))
        if self.lineEdit_timesteps.text()!='':
            if self.radioButton_timesteps1.isChecked() == True and (int(self.lineEdit_timesteps.text()) not in para.config.time_steps):
                para.config.time_steps.append(int(self.lineEdit_timesteps.text()))
            if self.radioButton_timesteps2.isChecked() == True:
                para.config.time_steps.remove(int(self.lineEdit_timesteps.text()))
            self.textBrowser_showtimesteps.setText(str(para.config.time_steps))
        if self.lineEdit_cellsize.text()!='':
            if self.radioButton_cellsize1.isChecked() == True and (int(self.lineEdit_cellsize.text()) not in para.config.cell_size):
                para.config.cell_size.append(int(self.lineEdit_cellsize.text()))
            if self.radioButton_cellsize2.isChecked() == True:
                para.config.cell_size.remove(int(self.lineEdit_cellsize.text()))
            self.textBrowser_showcellsize.setText(str(para.config.cell_size))
        if self.lineEdit_lr.text()!='':
            if self.radioButton_lr1.isChecked() == True and (float(self.lineEdit_lr.text()) not in para.config.lr):
                para.config.lr.append(float(self.lineEdit_lr.text()))
            if self.radioButton_lr2.isChecked() == True:
                para.config.lr.remove(float(self.lineEdit_lr.text()))
            self.textBrowser_showtlr.setText(str(para.config.lr))
        if self.lineEdit_iter.text()!='':
            if self.radioButton_iter1.isChecked() == True and (int(self.lineEdit_iter.text()) not in para.config.n_iter):
                para.config.n_iter.append(int(self.lineEdit_iter.text()))
            if self.radioButton_liter2.isChecked() == True:
                para.config.n_iter.remove(int(self.lineEdit_iter.text()))
            self.textBrowser_showiter.setText(str(para.config.n_iter))
        if self.lineEdit_keepprob_nn.text()!='':
            if self.radioButton_kpnn1.isChecked() == True and (float(self.lineEdit_keepprob_nn.text()) not in para.config.keep_prob_nn):
                para.config.keep_prob_nn.append(float(self.lineEdit_keepprob_nn.text()))
            if self.radioButton_kpnn2.isChecked() == True:
                para.config.keep_prob_nn.remove(float(self.lineEdit_keepprob_nn.text()))
            self.textBrowser_showkpnn.setText(str(para.config.keep_prob_nn))
        if self.lineEdit_kplstm.text()!='':
            if self.radioButton_kplstm1.isChecked() == True and (float(self.lineEdit_kplstm.text()) not in para.config.keep_prob_lstm):
                para.config.keep_prob_lstm.append(float(self.lineEdit_kplstm.text()))
            if self.radioButton_lkplstm2.isChecked() == True:
                para.config.keep_prob_lstm.remove(float(self.lineEdit_kplstm.text()))
            self.textBrowser_keep_prob_lstm.setText(str(para.config.keep_prob_lstm))
        if self.lineEdit_nn_layer.text()!='':
            if self.radioButton_nn_layer1.isChecked() == True and (int(self.lineEdit_nn_layer.text()) not in para.config.nn_layer):
                para.config.nn_layer.append(int(self.lineEdit_nn_layer.text()))
            if self.radioButton_nn_layer2.isChecked() == True:
                para.config.nn_layer.remove(int(self.lineEdit_nn_layer.text()))
            self.textBrowser_show_nn_layer.setText(str(para.config.nn_layer))
        if self.lineEdit_lstm_layer.text()!='':
            if self.radioButton_lstm_layer1.isChecked() == True and (int(self.lineEdit_lstm_layer.text()) not in para.config.lstm_layer):
                para.config.lstm_layer.append(int(self.lineEdit_lstm_layer.text()))
            if self.radioButton_lstm_layer2.isChecked() == True:
                para.config.lstm_layer.remove(int(self.lineEdit_lstm_layer.text()))
            self.textBrowser_showlstm_layer.setText(str(para.config.lstm_layer))
        if self.lineEdit_lstm_output.text()!='':
            if self.radioButton_lstm_output1.isChecked() == True and (int(self.lineEdit_lstm_output.text()) not in para.config.lstm_output):
                para.config.lstm_output.append(int(self.lineEdit_lstm_output.text()))
            if self.radioButton_lstm_output2.isChecked() == True:
                para.config.lstm_output.remove(int(self.lineEdit_lstm_output.text()))
            self.textBrowser_showlstm_output.setText(str(para.config.lstm_output))
        if self.lineEdit_nn_cellsize.text()!='':
            if self.radioButton_nn_cellsize1.isChecked() == True and (int(self.lineEdit_nn_cellsize.text()) not in para.config.nn_cellsize):
                para.config.nn_cellsize.append(int(self.lineEdit_nn_cellsize.text()))
            if self.radioButton_nn_cellsize2.isChecked() == True:
                para.config.nn_cellsize.remove(int(self.lineEdit_nn_cellsize.text()))
            self.textBrowser_show_nn_cellsize.setText(str(para.config.nn_cellsize))
    def click_back_to_hands(self):
        self.setVisible(False)
        ui.main_win.setVisible(True)

    def click_begin_auto(self):
        if self.comboBox_data.currentText() == "同花顺":
            data = pd.read_csv("dataset/tonghuashun1.csv")
        elif self.comboBox_data.currentText() == "茅台":
            data = pd.read_csv("dataset/maotai.csv")
        elif self.comboBox_data.currentText() == "万科":
            data = pd.read_csv("dataset/wanke.csv")
        elif self.comboBox_data.currentText() == "振业":
            data = pd.read_csv("dataset/深振业A.csv")
        elif self.comboBox_data.currentText() == "原野":
            data = pd.read_csv("dataset/深原野A.csv")
        elif self.comboBox_data.currentText() == "锦兴":
            data = pd.read_csv("dataset/深锦兴A.csv")
        elif self.comboBox_data.currentText() == "金田":
            data = pd.read_csv("dataset/深金田A.csv")
        elif self.comboBox_data.currentText() == "发展":
            data = pd.read_csv("dataset/深发展A.csv")
        elif self.comboBox_data.currentText() == "达声":
            data = pd.read_csv("dataset/深达声A.csv")
        elif self.comboBox_data.currentText() == "宝安":
            data = pd.read_csv("dataset/深宝安A.csv")
        elif self.comboBox_data.currentText() == "安达":
            data = pd.read_csv("dataset/深安达A.csv")

        best_config, all_config = auto.auto_func(xdata=data)
        self.textBrowser.append("本次自动调参最好参数为：")
        self.textBrowser.append(str(best_config))
        #self.textBrowser.append("最好参数训练模型的测试集准确率：")
        #self.textBrowser.append(str(para.config.acc))
        self.textBrowser.append("本次自动调参全部参数：")
        self.textBrowser.append(str(all_config))
        self.textBrowser.append("本次自动调参完成")
        self.textBrowser.append("***************************")

        savingpath = self.lineEdit_save.text()
        restoringpath = self.lineEdit_restore.text()

        self.f_temp = MyFigure()
        self.f_auto = MyFigure()
        self.f_temp.plotcost(itera=best_config['N_ITER'],lr=best_config['LR'], batch_size=best_config['BATCH_SIZE'],
                        timesteps=best_config['TIME_STEPS'], lstm_cell=best_config['CELL_SIZE'], lstm_layer=best_config['LSTM_LAYER'], nn_layer=best_config['NN_LAYER'],
                             is_train=True,lstm_keep_prob=best_config['KEEP_PROB_LSTM'],
                        nn_keep_prob=best_config['KEEP_PROB_NN'], savingpath=savingpath, restoringpath=restoringpath,
                        data=data,lstm_output=best_config['LSTM_OUTPUT'],nn_cellsize=best_config['NN_CELLSIZE'])
        self.f_auto.plotpred(lr=best_config['LR'], batch_size=best_config['BATCH_SIZE'],
                        timesteps=best_config['TIME_STEPS'], lstm_cell=best_config['CELL_SIZE'], lstm_layer=best_config['LSTM_LAYER'], nn_layer=best_config['NN_LAYER'],
                             is_test=True,restoringpath=restoringpath,
                        data=data,lstm_output=best_config['LSTM_OUTPUT'],nn_cellsize=best_config['NN_CELLSIZE'])
        if self.count_pred == 0:
            self.gridlayout_pred = QGridLayout(self.groupBox)  # 继承容器groupBox
            self.gridlayout_pred.addWidget(self.f_auto, 0, 1)
        else:
            self.gridlayout_pred.addWidget(self.f_auto, 0, 1)
        self.count_pred = self.count_pred + 1


class Mywindowpred(QMainWindow,pred.Ui_Form):
    def __init__(self,parent=None):
        super(Mywindowpred, self).__init__(parent)
        self.setupUi(self)
        self.pushButton_back.clicked.connect(self.click_back)
        self.pushButton_pred.clicked.connect(self.click_pred)
        self.lineEdit_batch.setText("50")
        self.lineEdit_timestep.setText("20")
        self.lineEdit_lstmlayer.setText(("3"))
        self.lineEdit_cellsize.setText("14")
        self.lineEdit_datapath.setText("dataset/tonghuashun1.csv")
        self.lineEdit_restorepath.setText("test/")
        self.lineEdit_lstmoutput.setText("1")
        self.lineEdit_nn_cellsize.setText("30")
        self.setStyleSheet("#Form{border-image:url(image/bg1.jpg);}")
        self.lineEdit_nn_layer.setText("1")

    def click_back(self):
        self.setVisible(False)
        ui.main_win.setVisible(True)
    def click_pred(self):
        data_path = self.lineEdit_datapath.text()
        data = pd.read_csv(data_path)
        batch_size = int(self.lineEdit_batch.text())
        timesteps = int(self.lineEdit_timestep.text())
        cellsize = int(self.lineEdit_cellsize.text())
        lstm_layer = int(self.lineEdit_lstmlayer.text())
        nn_layer = int(self.lineEdit_nn_layer.text())
        restorepath = self.lineEdit_restorepath.text()
        nn_cellsize = int(self.lineEdit_nn_cellsize.text())
        lstm_output = int(self.lineEdit_lstmoutput.text())
        pred_temp = p.fun(TIME_STEPS=timesteps,BATCH_SIZE=batch_size,CELL_SIZE=cellsize,LSTM_LAYER=lstm_layer,
                          NN_LAYER=nn_layer,restorePath=restorepath,data=data,NN_CELLSIZE=nn_cellsize,LSTM_OUTPUT=lstm_output)
        self.label_show.setText(str(pred_temp))


class MyWindow(QMainWindow,myui.Ui_MainWindow):
    def __init__(self,parent=None):
        super(MyWindow,self).__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(QIcon('stock_icon.png'))
        self.setWindowTitle('股票预测————基于LSTM神经网络')
        self.setStyleSheet("#MainWindow{border-image:url(image/bg1.jpg);}")


        self.pushButton.clicked.connect(self.click_start)
        self.pushButton_3.clicked.connect(self.click_start2)
        self.pushButton_toAuto.clicked.connect(self.click_toAuto)
        self.pushButton_pred.clicked.connect(self.click_topred)
        self.comboBox_dataset.currentIndexChanged.connect(self.click_combobox)
        self.comboBox_dataset_3.currentIndexChanged.connect(self.click_combobox3)
        self.pushButton_df.clicked.connect(self.click_todf)
        self.pushButton_recommend.clicked.connect(self.click_torecommend)

        self.automl = MyWinodw2()
        self.predwindow = Mywindowpred()
        self.dataframe = Mydataframe()
        self.recommendform = MyRecmendation()

        #初始化lstm_nn
        self.lineEdit_save.setText('test/model.ckpt')
        self.lineEdit_restore.setText('test/')
        self.lineEdit_iter.setText('5')
        self.lineEdit_timestep.setText('20')
        self.lineEdit_lr.setText('0.006')
        self.lineEdit_batchsize.setText('50')
        self.lineEdit_nn_keepprob.setText('0.8')
        self.lineEdit_lstm_keepprob.setText("0.8")
        self.lineEdit_lstm_cell.setText('14')
        self.lineEdit_lstm_layer.setText('3')
        self.checkBox_train.setCheckState(2)
        self.checkBox_predict.setCheckState(2)
        #初始化lstm
        self.lineEdit_save_3.setText('test/model.ckpt')
        self.lineEdit_restore_3.setText('test/')
        self.lineEdit_iter_3.setText('5')
        self.lineEdit_timestep_3.setText('20')
        self.lineEdit_lr_3.setText('0.006')
        self.lineEdit_batchsize_3.setText('50')
        self.lineEdit_lstm_keepprob_3.setText("0.8")
        self.lineEdit_lstm_cell_3.setText('14')
        self.lineEdit_lstm_layer_3.setText('3')
        self.checkBox_train_3.setCheckState(2)
        self.checkBox_predict_3.setCheckState(2)

        self.count_cost = 0
        self.count_pred = 0
        self.count_cost2 = 0
        self.count_pred2 = 0


    def click_torecommend(self):
        self.setVisible(False)
        self.recommendform.setVisible(True)

    def click_todf(self):
        self.setVisible(False)
        self.dataframe.setVisible(True)

    def click_combobox3(self):
        is_open = self.checkBox_openPrice.isChecked()
        if self.comboBox_dataset_3.currentText() == "同花顺":
            self.label_stockkind.setText("创业板")
            self.label_stockid.setText("300033")
            self.label_stockname.setText("同花顺")
        if self.comboBox_dataset_3.currentText() == "茅台":
            self.label_stockkind.setText("沪A")
            self.label_stockid.setText("600519")
            self.label_stockname.setText("贵州茅台")
        if self.comboBox_dataset_3.currentText() == "万科":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000002")
            self.label_stockname.setText("万科")
        if self.comboBox_dataset_3.currentText() == "振业":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000006")
            self.label_stockname.setText("振业")
        if self.comboBox_dataset_3.currentText() == "原野":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000005")
            self.label_stockname.setText("世纪星源")
        if self.comboBox_dataset_3.currentText() == "锦兴":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000008")
            self.label_stockname.setText("神州高铁")
        if self.comboBox_dataset_3.currentText() == "金田":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000003")
            self.label_stockname.setText("PT金田")
        if self.comboBox_dataset_3.currentText() == "发展":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000001")
            self.label_stockname.setText("平安银行")
        if self.comboBox_dataset_3.currentText() == "达声":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000007")
            self.label_stockname.setText("全新好")
        if self.comboBox_dataset_3.currentText() == "宝安":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000009")
            self.label_stockname.setText("中国宝安")
        if self.comboBox_dataset_3.currentText() == "安达":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000004")
            self.label_stockname.setText("国农科技")
        if is_open:
            self.label_predkind.setText("开盘价")
        else:
            self.label_predkind.setText("收盘价")
    def click_combobox(self):
        is_open = self.checkBox_openPrice.isChecked()
        if self.comboBox_dataset.currentText()=="同花顺":
            self.label_stockkind.setText("创业板")
            self.label_stockid.setText("300033")
            self.label_stockname.setText("同花顺")
        if self.comboBox_dataset.currentText()=="茅台":
            self.label_stockkind.setText("沪A")
            self.label_stockid.setText("600519")
            self.label_stockname.setText("贵州茅台")
        if self.comboBox_dataset.currentText()=="万科":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000002")
            self.label_stockname.setText("万科")
        if self.comboBox_dataset.currentText()=="振业":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000006")
            self.label_stockname.setText("振业")
        if self.comboBox_dataset.currentText()=="原野":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000005")
            self.label_stockname.setText("世纪星源")
        if self.comboBox_dataset.currentText()=="锦兴":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000008")
            self.label_stockname.setText("神州高铁")
        if self.comboBox_dataset.currentText()=="金田":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000003")
            self.label_stockname.setText("PT金田")
        if self.comboBox_dataset.currentText()=="发展":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000001")
            self.label_stockname.setText("平安银行")
        if self.comboBox_dataset.currentText()=="达声":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000007")
            self.label_stockname.setText("全新好")
        if self.comboBox_dataset.currentText()=="宝安":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000009")
            self.label_stockname.setText("中国宝安")
        if self.comboBox_dataset.currentText()=="安达":
            self.label_stockkind.setText("深A")
            self.label_stockid.setText("000004")
            self.label_stockname.setText("国农科技")
        if is_open:
            self.label_predkind.setText("开盘价")
        else:
            self.label_predkind.setText("收盘价")

    def click_toAuto(self):
        self.setVisible(False)
        self.automl.setVisible(True)


    def click_topred(self):
        self.predwindow.setVisible(True)
        self.setVisible(False)

    def check_null(self):
        if self.lineEdit_lstm_layer.text() == '' or self.lineEdit_lstm_cell.text() == '' \
                or self.lineEdit_lstm_keepprob.text() == '' or self.lineEdit_nn_keepprob.text() == '' \
                or self.lineEdit_batchsize.text() == '' or self.lineEdit_timestep.text() == '' or \
                self.lineEdit_lr.text() == '' or self.lineEdit_iter.text() == '' or self.lineEdit_nn_cell.text() == ''\
                or self.lineEdit_lstm_output == '':
            return True
        else:
            return False



    def click_start(self):
        is_checked = self.check_null()
        if is_checked:
            QMessageBox().warning(self,'警告','请输入所需参数以调试模型!',QMessageBox.Ok)
        else:
            try:
                itera = int(self.lineEdit_iter.text())
                lr = float(self.lineEdit_lr.text())
                batch_size = int(self.lineEdit_batchsize.text())
                timesteps = int(self.lineEdit_timestep.text())
                lstm_cell = int(self.lineEdit_lstm_cell.text())
                lstm_layer = int(self.lineEdit_lstm_layer.text())
                nn_layer = int(self.comboBox_nn_layer.currentText())
                is_train = bool(self.checkBox_train.isChecked())
                is_test = bool(self.checkBox_predict.isChecked())
                lstm_keep_prob = float(self.lineEdit_lstm_keepprob.text())
                nn_keep_prob = float(self.lineEdit_nn_keepprob.text())
                savingpath = self.lineEdit_save.text()
                restoringpath = self.lineEdit_restore.text()
                lstm_output = int(self.lineEdit_lstm_output.text())
                nn_cellsize = int(self.lineEdit_nn_cell.text())
            except:
                QMessageBox.critical(self, '错误', '请重新输入!', QMessageBox.Ok)
            else:
                if self.comboBox_dataset.currentText()=="同花顺":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/tonghuashun.csv")
                    else:
                        data = pd.read_csv("dataset/tonghuashun1.csv")
                elif self.comboBox_dataset.currentText()=="茅台":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/maotai.csv")
                    else:
                        data = pd.read_csv("dataset/maotai.csv")
                elif self.comboBox_dataset.currentText()=="万科":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/wanke.csv")
                    else:
                        data = pd.read_csv("dataset/wanke.csv")
                elif self.comboBox_dataset.currentText()=="振业":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深振业A.csv")
                    else:
                        data = pd.read_csv("dataset/深振业A.csv")
                elif self.comboBox_dataset.currentText()=="原野":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深原野A.csv")
                    else:
                        data = pd.read_csv("dataset/深原野A.csv")
                elif self.comboBox_dataset.currentText()=="锦兴":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深锦兴A.csv")
                    else:
                        data = pd.read_csv("dataset/深锦兴A.csv")
                elif self.comboBox_dataset.currentText()=="金田":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深金田A.csv")
                    else:
                        data = pd.read_csv("dataset/深金田A.csv")
                elif self.comboBox_dataset.currentText()=="发展":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深发展A.csv")
                    else:
                        data = pd.read_csv("dataset/深发展A.csv")
                elif self.comboBox_dataset.currentText()=="达声":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深达声A.csv")
                    else:
                        data = pd.read_csv("dataset/深达声A.csv")
                elif self.comboBox_dataset.currentText()=="宝安":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深宝安A.csv")
                    else:
                        data = pd.read_csv("dataset/深宝安A.csv")
                elif self.comboBox_dataset.currentText()=="安达":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深安达A.csv")
                    else:
                        data = pd.read_csv("dataset/深安达A.csv")
                elif self.comboBox_dataset.currentText()=="恒邦股份":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/hengbang.csv")
                    else:
                        data = pd.read_csv("dataset/hengbang.csv")
                elif self.comboBox_dataset.currentText()=="山东黄金":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/shandonghuangjin.csv")
                    else:
                        data = pd.read_csv("dataset/shandonghuangjin.csv")

                if is_train == True:
                    if self.lineEdit_save.text() == '':
                        QMessageBox.warning(self, "警告", "请输入模型参数存储路径", QMessageBox.Ok)
                    else:
                        self.f_cost = MyFigure()
                        self.f_cost.plotcost(itera=itera, lr=lr, batch_size=batch_size,
                                    timesteps=timesteps, lstm_cell=lstm_cell, lstm_layer=lstm_layer,nn_layer=nn_layer,
                                    is_train=is_train, lstm_keep_prob=lstm_keep_prob,
                                    nn_keep_prob=nn_keep_prob, savingpath=savingpath, restoringpath=restoringpath,
                             data=data, nn_cellsize=nn_cellsize, lstm_output=lstm_output)
                        if self.count_cost == 0:
                            self.gridLayout_cost = QGridLayout(self.groupBox1_cost)
                            self.gridLayout_cost.addWidget(self.f_cost,0,1)
                        else:
                            self.gridLayout_cost.addWidget(self.f_cost, 0, 1)
                        self.count_cost = self.count_cost + 1

                if is_test == True:
                    if self.lineEdit_save.text() == '':
                        QMessageBox.warning(self, "警告", "请输入模型参数读取路径", QMessageBox.Ok)
                    else:
                        self.f = MyFigure()
                        self.f.plotpred(lr=lr, batch_size=batch_size,
                                        timesteps=timesteps, lstm_cell=lstm_cell, lstm_layer=lstm_layer,nn_layer=nn_layer,
                                        is_test=is_test, restoringpath=restoringpath,
                                 data=data,nn_cellsize=nn_cellsize, lstm_output=lstm_output)
                        if self.count_pred == 0:
                            self.gridlayout_pred = QGridLayout(self.groupBox1_pred)  # 继承容器groupBox
                            self.gridlayout_pred.addWidget(self.f, 0, 1)
                        else:
                            self.gridlayout_pred.addWidget(self.f, 0, 1)
                        self.count_pred = self.count_pred + 1

                        self.label.clear()
                        self.label.setFont(QFont("Microsoft YaHei"))
                        self.label.setStyleSheet("color:white")
                        self.label.setText("部分训练样例训练偏差："+str(self.f.acc1)+"\n测试样例偏差："+str(self.f.acc2)+"\n总偏差："+str(self.f.acc3))
    def check_null2(self):
        if self.lineEdit_lstm_layer_3.text() == '' or self.lineEdit_lstm_cell_3.text() == '' \
                or self.lineEdit_lstm_keepprob_3.text() == ''or self.lineEdit_batchsize.text() == '' \
                or self.lineEdit_timestep_3.text() == '' or \
                self.lineEdit_lr_3.text() == '' or self.lineEdit_iter_3.text() == '':
            return True
        else:
            return False

    def click_start2(self):
        is_checked2 = self.check_null2()
        if is_checked2:
            QMessageBox().warning(self, '警告', '请输入所需参数以调试模型!', QMessageBox.Ok)
        else:
            try:
                itera = int(self.lineEdit_iter_3.text())
                lr = float(self.lineEdit_lr_3.text())
                batch_size = int(self.lineEdit_batchsize_3.text())
                timesteps = int(self.lineEdit_timestep_3.text())
                lstm_cell = int(self.lineEdit_lstm_cell_3.text())
                lstm_layer = int(self.lineEdit_lstm_layer_3.text())
                is_train = bool(self.checkBox_train_3.isChecked())
                is_test = bool(self.checkBox_predict_3.isChecked())
                lstm_keep_prob = float(self.lineEdit_lstm_keepprob_3.text())
                savingpath = self.lineEdit_save.text()
                restoringpath = self.lineEdit_restore.text()
            except:
                QMessageBox.critical(self, '错误', '请重新输入!', QMessageBox.Ok)
            else:
                if self.comboBox_dataset_3.currentText() == "同花顺":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/tonghuashun.csv")
                    else:
                        data = pd.read_csv("dataset/tonghuashun1.csv")
                elif self.comboBox_dataset_3.currentText() == "茅台":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/maotai.csv")
                    else:
                        data = pd.read_csv("dataset/maotai.csv")
                elif self.comboBox_dataset_3.currentText() == "万科":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/wanke.csv")
                    else:
                        data = pd.read_csv("dataset/wanke.csv")
                elif self.comboBox_dataset_3.currentText() == "振业":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深振业A.csv")
                    else:
                        data = pd.read_csv("dataset/深振业A.csv")
                elif self.comboBox_dataset_3.currentText() == "原野":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深原野A.csv")
                    else:
                        data = pd.read_csv("dataset/深原野A.csv")
                elif self.comboBox_dataset_3.currentText() == "锦兴":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深锦兴A.csv")
                    else:
                        data = pd.read_csv("dataset/深锦兴A.csv")
                elif self.comboBox_dataset_3.currentText() == "金田":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深金田A.csv")
                    else:
                        data = pd.read_csv("dataset/深金田A.csv")
                elif self.comboBox_dataset_3.currentText() == "发展":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深发展A.csv")
                    else:
                        data = pd.read_csv("dataset/深发展A.csv")
                elif self.comboBox_dataset_3.currentText() == "达声":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深达声A.csv")
                    else:
                        data = pd.read_csv("dataset/深达声A.csv")
                elif self.comboBox_dataset_3.currentText() == "宝安":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深宝安A.csv")
                    else:
                        data = pd.read_csv("dataset/深宝安A.csv")
                elif self.comboBox_dataset_3.currentText() == "安达":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/深安达A.csv")
                    else:
                        data = pd.read_csv("dataset/深安达A.csv")
                elif self.comboBox_dataset_3.currentText() == "恒邦股份":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/hengbang.csv")
                    else:
                        data = pd.read_csv("dataset/hengbang.csv")
                elif self.comboBox_dataset_3.currentText() == "山东黄金":
                    if self.checkBox_openPrice.isChecked():
                        data = pd.read_csv("dataset_openPrice/shandonghuangjin.csv")
                    else:
                        data = pd.read_csv("dataset/shandonghuangjin.csv")

                if is_train == True:
                    if self.lineEdit_save_3.text() == '':
                        QMessageBox.warning(self, "警告", "请输入模型参数存储路径", QMessageBox.Ok)
                    else:
                        self.f_cost2 = MyFigure()
                        self.f_cost2.plotcost2(itera=itera, lr=lr, batch_size=batch_size,
                                             timesteps=timesteps, lstm_cell=lstm_cell, lstm_layer=lstm_layer,
                                             is_train=is_train, is_test=is_test, lstm_keep_prob=lstm_keep_prob,
                                            savingpath=savingpath,
                                             restoringpath=restoringpath,
                                             data=data)
                        if self.count_cost2 == 0:
                            self.gridLayout_cost = QGridLayout(self.groupBox2_cost)
                            self.gridLayout_cost.addWidget(self.f_cost2, 0, 1)
                        else:
                            self.gridLayout_cost.addWidget(self.f_cost2, 0, 1)
                        self.count_cost2 = self.count_cost2 + 1


                if is_test == True:
                    if self.lineEdit_save_3.text() == '':
                        QMessageBox.warning(self, "警告", "请输入模型参数读取路径", QMessageBox.Ok)
                    else:
                        self.f1 = MyFigure()
                        self.f1.plotpred2(itera=itera, lr=lr, batch_size=batch_size,
                                        timesteps=timesteps, lstm_cell=lstm_cell, lstm_layer=lstm_layer,
                                        is_train=is_train, is_test=is_test, lstm_keep_prob=lstm_keep_prob,
                                        savingpath=savingpath, restoringpath=restoringpath,
                                        data=data)
                        if self.count_pred2 == 0:
                            self.gridlayout_pred = QGridLayout(self.groupBox2_pred)  # 继承容器groupBox
                            self.gridlayout_pred.addWidget(self.f1, 0, 1)
                        else:
                            self.gridlayout_pred.addWidget(self.f1, 0, 1)
                        self.count_pred2 = self.count_pred2 + 1

                        self.label_3.clear()
                        self.label_3.setFont(QFont("Microsoft YaHei"))
                        self.label_3.setStyleSheet("color:white")
                        self.label_3.setText(
                            "测试样例偏差：" + str(self.f1.acc33))
#"部分训练样例训练偏差：" + str(self.f1.acc11) + "\n测试样例偏差：" + str(self.f1.acc22) +
class mylogin(QDialog,log.Ui_Dialog_login):
    def __init__(self,parent=None):
        super(mylogin, self).__init__(parent)
        self.setupUi(self)
        self.main_win = MyWindow()
        self.pushButton_login.clicked.connect(self.click_login)
    def click_login(self):
        if (self.lineEdit_user.text()=="cza" and self.lineEdit_password.text()=="cza"):
            self.main_win.show()
            self.close()
        else:
            QMessageBox().warning(self, '警告', '请输入正确的账号密码!', QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('stock_icon.png'))
    ui = mylogin()
    ui.show()
    sys.exit(app.exec_())
