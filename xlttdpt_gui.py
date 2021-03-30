# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'xlttdpt_gui.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QLabel, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
##import cv2, imutils
import numpy as np
import matplotlib.pyplot as plt
from daphuongtien import run

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 619)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.OriginalImage = QtWidgets.QLabel(self.centralwidget)
        self.OriginalImage.setGeometry(QtCore.QRect(30, 100, 361, 361))
        self.OriginalImage.setScaledContents(True)
        self.OriginalImage.setAlignment(QtCore.Qt.AlignCenter)
        self.OriginalImage.setObjectName("OriginalImage")
        
        self.CompressionImage = QtWidgets.QLabel(self.centralwidget)
        self.CompressionImage.setGeometry(QtCore.QRect(420, 100, 331, 361))
        self.CompressionImage.setScaledContents(True)
        self.CompressionImage.setAlignment(QtCore.Qt.AlignCenter)
        self.CompressionImage.setObjectName("CompressionImage")
        
        self.chooseImage = QtWidgets.QPushButton(self.centralwidget)
        self.chooseImage.setGeometry(QtCore.QRect(130, 490, 151, 51))
        self.chooseImage.setObjectName("chooseImage")
        
        self.Compress = QtWidgets.QPushButton(self.centralwidget)
        self.Compress.setGeometry(QtCore.QRect(520, 490, 161, 51))
        self.Compress.setObjectName("Compress")
        
        self.nameOriginal = QtWidgets.QLabel(self.centralwidget)
        self.nameOriginal.setGeometry(QtCore.QRect(110, 50, 151, 31))
        self.nameOriginal.setAlignment(QtCore.Qt.AlignCenter)
        self.nameOriginal.setObjectName("nameOriginal")
        
        self.nameCompression = QtWidgets.QLabel(self.centralwidget)
        self.nameCompression.setGeometry(QtCore.QRect(510, 40, 161, 41))
        self.nameCompression.setObjectName("nameCompression")
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 28))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.chooseImage.clicked.connect(self.browseImage)
        self.Compress.clicked.connect(self.compressImage)
        
        #vbox = QVBoxLayout()
        #vbox.addWidget(self.chooseImage)
        
        #vbox.addWidget(self.OriginalImage)
        
        #self.setLayout(vbox)
        #self.show()
        
        ## image address 
        self.filename = None
        # will hold the temporary image for display
        self.tmp = None
        self.image = None

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.OriginalImage.setText(_translate("MainWindow", "TextLabel"))
        self.CompressionImage.setText(_translate("MainWindow", "TextLabel"))
        self.chooseImage.setText(_translate("MainWindow", "Browse Image"))
        self.Compress.setText(_translate("MainWindow", "Compress Image"))
        self.nameOriginal.setText(_translate("MainWindow", "Original Image"))
        self.nameCompression.setText(_translate("MainWindow", "Compression Image"))
        
    def browseImage(self):
        self.filename = QFileDialog.getOpenFileName(None, 'Open File', './', 'Image file(*.jpg *.gif *.png *.jpeg)')
        imagePath = self.filename[0]
        #global image
        #image = fname
        pixmap = QPixmap(imagePath)
        self.OriginalImage.setPixmap(QPixmap(pixmap))
        # self.resize(pixmap.width(), pixmap.height())

        image = pixmap.toImage()
        size = pixmap.size()
        str = image.bits().asstring(size.height() * size.width() * 3)
        self.tmp = np.fromstring(str, dtype=np.uint8).reshape((size.height(), size.width(), 3))
    
    def compressImage(self):
        image = run(self.tmp)
        QImg = QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888)
        
        #imageCompression = self.browseImage.pixmap
        imageCompressionPath = self.filename[0]
        pixmapCompress = QPixmap(QImg)
        self.CompressionImage.setPixmap(QPixmap(pixmapCompress))
        #self.resize(pixmapCompress.width(), pixmapCompress.height())
        
        
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

