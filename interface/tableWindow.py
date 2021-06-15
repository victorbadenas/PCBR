# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interface/tableWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd

class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        newdf = pd.DataFrame(columns=['ID'] + list(data.columns))
        data['ID'] = data.index
        for column in newdf.columns:
            newdf[column] = [column] + data[column].values.tolist()
        self._data = newdf.values.tolist()
        

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._data[0])

class TableWindow(object):
    def setupUi(self, Dialog, data:pd.DataFrame, text="", size=(850, 350), buttons=("No", "Yes"), title=None):
        for b in buttons:
            assert b in ['Yes', 'No', 'Ok'], f"{b} not in ['Yes', 'No', 'Ok']"
        if title is None:
            title = self.__class__.__name__

        self.result = None

        self.Dialog = Dialog
        self.Dialog.setObjectName("Dialog")
        self.Dialog.resize(*size)

        self.gridLayout = QtWidgets.QGridLayout(self.Dialog)
        self.gridLayout.setObjectName("gridLayout")

        self.label = QtWidgets.QLabel(self.Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)

        self.tableView = QtWidgets.QTableView(self.Dialog)
        self.tableModel = TableModel(data)
        self.tableView.setModel(self.tableModel)
        self.tableView.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        self.tableView.setObjectName("tableView")
        self.gridLayout.addWidget(self.tableView, 1, 0, 1, 2)

        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        # self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.No|QtWidgets.QDialogButtonBox.Yes)
        for b in buttons:
            if b == 'Yes':
                b = QtWidgets.QDialogButtonBox.Yes
                f = self.yes
            elif b == 'No':
                b = QtWidgets.QDialogButtonBox.No
                f = self.no
            if b == 'Ok':
                b = QtWidgets.QDialogButtonBox.Ok
                f = self.ok
            self.buttonBox.addButton(b)
            self.buttonBox.button(b).clicked.connect(f)

        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 2, 0, 1, 1)

        self.retranslateUi(text=text, title=title)
        QtCore.QMetaObject.connectSlotsByName(self.Dialog)

    def retranslateUi(self, text, title):
        _translate = QtCore.QCoreApplication.translate
        self.Dialog.setWindowTitle(_translate("Dialog", title))
        self.label.setText(_translate("Dialog", text))

    def yes(self):
        self.result = 'Yes'
        self.close()

    def no(self):
        self.result = 'No'
        self.close()

    def ok(self):
        self.result = 'Ok'
        self.close()

    def close(self):
        self.Dialog.close()

if __name__ == "__main__":
    # |                   | CPU                  |   RAM (GB) |   SSD (GB) |   HDD (GB) | GPU                    |   Optical Drive (1 = DVD; 0 = None) |   Price (€) |
    # |:------------------|:---------------------|-----------:|-----------:|-----------:|:-----------------------|------------------------------------:|------------:|
    # | Original solution | Intel Core i5-11600K |         32 |        250 |       1000 | Radeon RX 580          |                                   1 |      801.92 |
    # | 1                 | Intel Core i5-11600K |         32 |        250 |       1000 | GeForce GTX 1660 SUPER |                                   1 |      801.92 |
    # | 2                 | AMD Ryzen 9 5900X    |         32 |        250 |       1000 | GeForce GTX 1660 SUPER |                                   1 |     1190.92 |
    # | 3                 | AMD Ryzen 9 5900X    |         32 |        250 |       1000 | GeForce GTX 1660 SUPER |                                   1 |     1190.92 |
    # | 4                 | AMD Ryzen 9 5900X    |         32 |        250 |       1000 | GeForce GTX 1660 SUPER |                                   1 |     1190.92 |
    # | 5                 | AMD Ryzen 9 5900X    |         32 |        250 |       1000 | GeForce GTX 1660 SUPER |                                   1 |     1190.92 |
    import sys
    data = [
        ['ID','CPU','RAM (GB)','SSD (GB)','HDD (GB)','GPU','Optical Drive','Price (€)'],
        ["Original solution", "Intel Core i5-11600K", "32", "250", "1000", "Radeon RX 580", "1", "801.92"],
        ["1","Intel Core i5-11600K","32","250","1000","GeForce GTX 1660 SUPER","1","801.92"],
        ["2","AMD Ryzen 9 5900X","32","250","1000","GeForce GTX 1660 SUPER","1","1190.92"],
        ["3","AMD Ryzen 9 5900X","32","250","1000","GeForce GTX 1660 SUPER","1","1190.92"],
        ["4","AMD Ryzen 9 5900X","32","250","1000","GeForce GTX 1660 SUPER","1","1190.92"],
        ["5","AMD Ryzen 9 5900X","32","250","1000","GeForce GTX 1660 SUPER","1","1190.92"],
    ]
    data = pd.DataFrame(data[1:], columns=data[0])
    data.set_index('ID', inplace=True)
    data.index.rename('', inplace=True)
    print(data.to_markdown())
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = TableWindow()
    ui.setupUi(Dialog, data=data, text='this looks way better')
    Dialog.show()
    app.exec_()
    print(ui.result)
