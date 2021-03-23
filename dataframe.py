# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dataframe.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1280, 686)
        self.tableView = QtWidgets.QTableView(Form)
        self.tableView.setGeometry(QtCore.QRect(0, 290, 1281, 511))
        self.tableView.setObjectName("tableView")
        self.pushButton_back = QtWidgets.QPushButton(Form)
        self.pushButton_back.setGeometry(QtCore.QRect(1150, 10, 113, 32))
        self.pushButton_back.setObjectName("pushButton_back")
        self.comboBox_data = QtWidgets.QComboBox(Form)
        self.comboBox_data.setGeometry(QtCore.QRect(130, 20, 251, 26))
        self.comboBox_data.setObjectName("comboBox_data")
        self.comboBox_data.addItem("")
        self.comboBox_data.setItemText(0, "请选择")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.label_data = QtWidgets.QLabel(Form)
        self.label_data.setGeometry(QtCore.QRect(20, 20, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_data.setFont(font)
        self.label_data.setObjectName("label_data")
        self.groupBox_1 = QtWidgets.QGroupBox(Form)
        self.groupBox_1.setGeometry(QtCore.QRect(10, 40, 391, 241))
        self.groupBox_1.setObjectName("groupBox_1")
        self.groupBox_3 = QtWidgets.QGroupBox(Form)
        self.groupBox_3.setGeometry(QtCore.QRect(870, 40, 391, 241))
        self.groupBox_3.setObjectName("groupBox_3")
        self.groupBox_2 = QtWidgets.QGroupBox(Form)
        self.groupBox_2.setGeometry(QtCore.QRect(430, 40, 411, 241))
        self.groupBox_2.setObjectName("groupBox_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "查看数据集"))
        self.pushButton_back.setText(_translate("Form", "返回"))
        self.comboBox_data.setItemText(1, _translate("Form", "茅台"))
        self.comboBox_data.setItemText(2, _translate("Form", "同花顺"))
        self.comboBox_data.setItemText(3, _translate("Form", "万科"))
        self.comboBox_data.setItemText(4, _translate("Form", "振业"))
        self.comboBox_data.setItemText(5, _translate("Form", "原野"))
        self.comboBox_data.setItemText(6, _translate("Form", "锦兴"))
        self.comboBox_data.setItemText(7, _translate("Form", "金田"))
        self.comboBox_data.setItemText(8, _translate("Form", "发展"))
        self.comboBox_data.setItemText(9, _translate("Form", "达声"))
        self.comboBox_data.setItemText(10, _translate("Form", "宝安"))
        self.comboBox_data.setItemText(11, _translate("Form", "安达"))
        self.comboBox_data.setItemText(12, _translate("Form", "恒邦股份"))
        self.label_data.setText(_translate("Form", "选择数据集"))
        self.groupBox_1.setTitle(_translate("Form", "股价"))
        self.groupBox_3.setTitle(_translate("Form", "市值"))
        self.groupBox_2.setTitle(_translate("Form", "成交量"))

