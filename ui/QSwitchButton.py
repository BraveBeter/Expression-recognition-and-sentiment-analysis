from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class SwitchButton(QLabel):
    """自定义Switch按钮"""

    def __init__(self, parent=None):
        super(SwitchButton, self).__init__(parent)

        # 设置无边框和背景透明
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(50, 25)
        self.state = False  # 按钮状态：True表示开，False表示关

    def mousePressEvent(self, event):
        """鼠标点击事件：用于切换按钮状态"""
        super(SwitchButton, self).mousePressEvent(event)
        self.state = False if self.state else True
        # print(self.state)
        self.update()

    def paintEvent(self, event):
        """绘制按钮"""
        super(SwitchButton, self).paintEvent(event)

        # 创建绘制器并设置抗锯齿和图片流畅转换
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # 定义字体样式
        font = QFont('Microsoft YaHei')
        font.setPixelSize(14)
        painter.setFont(font)

        # 开关为开的状态
        if self.state:
            # 绘制背景
            painter.setPen(Qt.NoPen)
            brush = QBrush(QColor('#00c801'))
            painter.setBrush(brush)
            # painter.drawRoundedRect(0, 0, self.width(), self.height(), self.height() // 2, self.height() // 2)
            painter.drawRoundedRect(0, 0, 50, 26, 12, 12)

            # 绘制圆圈
            painter.setPen(Qt.NoPen)
            brush.setColor(QColor('#ffffff'))
            painter.setBrush(brush)
            painter.drawRoundedRect(23, 1, 24, 24, 12, 12)
        # 开关为关的状态
        else:
            # 绘制背景
            painter.setPen(Qt.NoPen)
            brush = QBrush(QColor('#e1e1e1'))
            painter.setBrush(brush)
            # painter.drawRoundedRect(0, 0, self.width(), self.height(), self.height()//2, self.height()//2)
            painter.drawRoundedRect(0, 0, 50, 26, 12, 12)

            painter.setPen(Qt.NoPen)
            brush.setColor(QColor('#ffffff'))
            painter.setBrush(brush)
            painter.drawRoundedRect(3, 1, 24, 24, 12, 12)


from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
# 确保导入了QSwitchButton类和其他必要的模块


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 创建QSwitchButton实例
        self.switchButton = SwitchButton(self)

        # 连接信号到槽函数
        # self.switchButton.clickedOn.connect(self.onSwitchedOn)
        # self.switchButton.clickedOff.connect(self.onSwitchedOff)

        # 设置窗口布局
        layout = QVBoxLayout()
        layout.addWidget(self.switchButton)

        # 创建一个容器widget并设置布局
        container = QWidget()
        container.setLayout(layout)

        # 设置主窗口的中心widget为容器
        self.setCentralWidget(container)

    # 定义槽函数
    def onSwitchedOn(self):
        print("Switch is On")

    def onSwitchedOff(self):
        print("Switch is Off")


# 创建应用程序实例和主窗口，然后运行
if __name__ == '__main__':
    app = QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_()