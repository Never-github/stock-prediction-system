# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'login.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog_login(object):
    def setupUi(self, Dialog_login):
        Dialog_login.setObjectName("Dialog_login")
        Dialog_login.setWindowModality(QtCore.Qt.NonModal)
        Dialog_login.resize(400, 300)
        self.lineEdit_user = QtWidgets.QLineEdit(Dialog_login)
        self.lineEdit_user.setGeometry(QtCore.QRect(170, 80, 151, 21))
        self.lineEdit_user.setObjectName("lineEdit_user")
        self.label_user = QtWidgets.QLabel(Dialog_login)
        self.label_user.setGeometry(QtCore.QRect(70, 80, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_user.setFont(font)
        self.label_user.setObjectName("label_user")
        self.label_user_2 = QtWidgets.QLabel(Dialog_login)
        self.label_user_2.setGeometry(QtCore.QRect(70, 130, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_user_2.setFont(font)
        self.label_user_2.setObjectName("label_user_2")
        self.lineEdit_password = QtWidgets.QLineEdit(Dialog_login)
        self.lineEdit_password.setGeometry(QtCore.QRect(170, 130, 151, 21))
        self.lineEdit_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_password.setObjectName("lineEdit_password")
        self.pushButton_login = QtWidgets.QPushButton(Dialog_login)
        self.pushButton_login.setGeometry(QtCore.QRect(80, 200, 91, 31))
        self.pushButton_login.setObjectName("pushButton_login")
        self.pushButton_exit = QtWidgets.QPushButton(Dialog_login)
        self.pushButton_exit.setGeometry(QtCore.QRect(210, 200, 91, 31))
        self.pushButton_exit.setObjectName("pushButton_exit")

        self.retranslateUi(Dialog_login)
        QtCore.QMetaObject.connectSlotsByName(Dialog_login)

    def retranslateUi(self, Dialog_login):
        _translate = QtCore.QCoreApplication.translate
        Dialog_login.setWindowTitle(_translate("Dialog_login", "登陆"))
        self.label_user.setText(_translate("Dialog_login", "用户:"))
        self.label_user_2.setText(_translate("Dialog_login", "密码:"))
        self.pushButton_login.setText(_translate("Dialog_login", "登陆"))
        self.pushButton_exit.setText(_translate("Dialog_login", "退出"))

