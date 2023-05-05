from PyQt5.QtWidgets import *


class ReconstructionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel("3D Reconstruction Tab"))
        self.setLayout(self.layout)
