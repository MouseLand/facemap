from PyQt5.QtWidgets import *

class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel("Analysis Tab"))
        self.setLayout(self.layout)