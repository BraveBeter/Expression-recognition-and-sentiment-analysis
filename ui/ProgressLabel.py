from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen
from PyQt5.QtCore import Qt
import sys

class ProgressBarLabel(QLabel):
    def __init__(self, parent=None):
        super(ProgressBarLabel, self).__init__(parent)
        self.progress = 0
        self.setMaximumHeight(22)
        self.setMaximumWidth(120)

    def setProgress(self, value):
        self.progress = value
        self.repaint()  # 触发重绘事件，更新进度条

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.transparent)  # 创建一个透明的画笔
        painter.setPen(pen)  # 设置画笔为透明
        painter.setBrush(QColor(215, 215, 215))  # 设置画刷为灰色
        painter.drawRect(0, 0, self.width(), self.height())  # 绘制灰色背景

        if self.progress > 0:
            painter.setBrush(QColor(125, 225, 105))  # 设置画刷为绿色
            painter.drawRect(0, 0, (self.width() * self.progress) // 100, self.height())  # 绘制绿色进度条

if __name__ == '__main__':
    app = QApplication(sys.argv)
    label = ProgressBarLabel()
    label.show()
    label.setProgress(85)  # 设置进度为50%
    sys.exit(app.exec_())
