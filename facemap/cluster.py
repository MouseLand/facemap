import umap
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
#import pyqtgraph.opengl as gl
from sklearn.cluster import MiniBatchKMeans
import hdbscan
from matplotlib import cm
from . import utils
from .gui import io
import cv2
import os
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QLabel, QPushButton, QRadioButton, QSpinBox, QButtonGroup,
        QMessageBox, QLineEdit, QCheckBox)

colors_list = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
        "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
        "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
        "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
        "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
        "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
        "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
        "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
        "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",
        "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
        "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
        "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
        "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
        "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
        "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
        "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
        "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
        "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
        "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B", "#1E2324", "#DEC9B2", "#9D4948",
        "#85ABB4", "#342142", "#D09685", "#A4ACAC", "#00FFFF", "#AE9C86", "#742A33", "#0E72C5",
        "#AFD8EC", "#C064B9", "#91028C", "#FEEDBF", "#FFB789", "#9CB8E4", "#AFFFD1", "#2A364C",
        "#4F4A43", "#647095", "#34BBFF", "#807781", "#920003", "#B3A5A7", "#018615", "#F1FFC8",
        "#976F5C", "#FF3BC1", "#FF5F6B", "#077D84", "#F56D93", "#5771DA", "#4E1E2A", "#830055",
        "#02D346", "#BE452D", "#00905E", "#BE0028", "#6E96E3", "#007699", "#FEC96D", "#9C6A7D",
        "#3FA1B8", "#893DE3", "#79B4D6", "#7FD4D9", "#6751BB", "#B28D2D", "#E27A05", "#DD9CB8",
        "#AABC7A", "#980034", "#561A02", "#8F7F00", "#635000", "#CD7DAE", "#8A5E2D", "#FFB3E1",
        "#6B6466", "#C6D300", "#0100E2", "#88EC69", "#8FCCBE", "#21001C", "#511F4D", "#E3F6E3",
        "#FF8EB1", "#6B4F29", "#A37F46", "#6A5950", "#1F2A1A", "#04784D", "#101835", "#E6E0D0",
        "#FF74FE", "#00A45F", "#8F5DF8", "#4B0059", "#412F23", "#D8939E", "#DB9D72", "#604143",
        "#B5BACE", "#989EB7", "#D2C4DB", "#A587AF", "#77D796", "#7F8C94", "#FF9B03", "#555196",
        "#31DDAE", "#74B671", "#802647", "#2A373F", "#014A68", "#696628", "#4C7B6D", "#002C27",
        "#7A4522", "#3B5859", "#E5D381", "#FFF3FF", "#679FA0", "#261300", "#2C5742", "#9131AF",
        "#AF5D88", "#C7706A", "#61AB1F", "#8CF2D4", "#C5D9B8", "#9FFFFB", "#BF45CC", "#493941",
        "#863B60", "#B90076", "#003177", "#C582D2", "#C1B394", "#602B70", "#887868", "#BABFB0",
        "#030012", "#D1ACFE", "#7FDEFE", "#4B5C71", "#A3A097", "#E66D53", "#637B5D", "#92BEA5",
        "#00F8B3", "#BEDDFF", "#3DB5A7", "#DD3248", "#B6E4DE", "#427745", "#598C5A", "#B94C59",
        "#8181D5", "#94888B", "#FED6BD", "#536D31", "#6EFF92", "#E4E8FF", "#20E200", "#FFD0F2",
        "#4C83A1", "#BD7322", "#915C4E", "#8C4787", "#025117", "#A2AA45", "#2D1B21", "#A9DDB0",
        "#FF4F78", "#528500", "#009A2E", "#17FCE4", "#71555A", "#525D82", "#00195A", "#967874",
        "#555558", "#0B212C", "#1E202B", "#EFBFC4", "#6F9755", "#6F7586", "#501D1D", "#372D00",
        "#741D16", "#5EB393", "#B5B400", "#DD4A38", "#363DFF", "#AD6552", "#6635AF", "#836BBA",
        "#98AA7F", "#464836", "#322C3E", "#7CB9BA", "#5B6965", "#707D3D", "#7A001D", "#6E4636",
        "#443A38", "#AE81FF", "#489079", "#897334", "#009087", "#DA713C", "#361618", "#FF6F01",
        "#006679", "#370E77", "#4B3A83", "#C9E2E6", "#C44170", "#FF4526", "#73BE54", "#C4DF72",
        "#ADFF60", "#00447D", "#DCCEC9", "#BD9479", "#656E5B", "#EC5200", "#FF6EC2", "#7A617E",
        "#DDAEA2", "#77837F", "#A53327", "#608EFF", "#B599D7", "#A50149", "#4E0025", "#C9B1A9",
        "#03919A", "#1B2A25", "#E500F1", "#982E0B", "#B67180", "#E05859", "#006039", "#578F9B",
        "#305230", "#CE934C", "#B3C2BE", "#C0BAC0", "#B506D3", "#170C10", "#4C534F", "#224451",
        "#3E4141", "#78726D", "#B6602B", "#200441", "#DDB588", "#497200", "#C5AAB6", "#033C61",
        "#71B2F5", "#A9E088", "#4979B0", "#A2C3DF"]

class Cluster():
    def __init__(self, parent, method=None, cluster_labels=None, cluster_labels_method=None, data_type=None):
        self.cluster_method = method
        self.cluster_labels = cluster_labels
        self.cluster_labels_method = cluster_labels_method
        self.data_type = data_type
        self.n_neighbors = None
        self.min_dist = None
        self.n_components = None
        self.embedded_output = None
        self.create_clustering_widgets(parent)

    def create_clustering_widgets(self, parent):
        # Add options to change params for embedding using user input
        parent.ClusteringLabel = QLabel("Clustering")
        parent.ClusteringLabel.setStyleSheet("color: white;")
        parent.ClusteringLabel.setAlignment(QtCore.Qt.AlignCenter)
        parent.ClusteringLabel.setFont(QFont("Arial", 12, QFont.Bold))

        parent.min_dist_label = QLabel("min_dist:")
        parent.min_dist_label.setStyleSheet("color: gray;")
        parent.min_dist_value = QLineEdit()
        parent.min_dist_value.setText(str(0.5))
        parent.min_dist_value.setFixedWidth(50)

        parent.n_neighbors_label = QLabel("n_neighbors:")
        parent.n_neighbors_label.setStyleSheet("color: gray;")
        parent.n_neighbors_value = QLineEdit()
        parent.n_neighbors_value.setText(str(30))
        parent.n_neighbors_value.setFixedWidth(50)

        parent.n_components_label = QLabel("n_components:")
        parent.n_components_label.setStyleSheet("color: gray;")
        parent.n_components_value = QSpinBox()
        parent.n_components_value.setRange(2, 3)
        parent.n_components_value.setValue(2)
        parent.n_components_value.setFixedWidth(50)
        #metric

        parent.cluster_method_label = QLabel("Cluster labels")
        parent.cluster_method_label.setStyleSheet("color: gray;")
        parent.cluster_method_label.setAlignment(QtCore.Qt.AlignCenter)

        parent.load_umap_embedding_button = QPushButton('Load emmbedding')

        parent.RadioGroup = QButtonGroup()
        parent.load_cluster_labels_button = QPushButton('Load')
        parent.loadlabels_radiobutton = QRadioButton("Load labels")
        parent.loadlabels_radiobutton.setStyleSheet("color: gray;")
        parent.RadioGroup.addButton(parent.loadlabels_radiobutton)
        parent.kmeans_radiobutton = QRadioButton("KMeans")
        parent.kmeans_radiobutton.setStyleSheet("color: gray;")
        parent.RadioGroup.addButton(parent.kmeans_radiobutton)
        parent.hdbscan_radiobutton = QRadioButton("HDBSCAN")
        parent.hdbscan_radiobutton.setStyleSheet("color: gray;")
        parent.RadioGroup.addButton(parent.hdbscan_radiobutton)

        parent.min_cluster_size_label = QLabel("min_cluster_size:")
        parent.min_cluster_size_label.setStyleSheet("color: gray;")
        parent.min_cluster_size = QLineEdit()
        parent.min_cluster_size.setFixedWidth(50)
        parent.min_cluster_size.setText(str(500))

        parent.num_clusters_label = QLabel("num_clusters:")
        parent.num_clusters_label.setStyleSheet("color: gray;")
        parent.num_clusters = QLineEdit()
        parent.num_clusters.setFixedWidth(50)
        parent.num_clusters.setText(str(5))

        istretch = 12
        parent.l0.addWidget(parent.ClusteringLabel, istretch, 0, 1, 2)
        parent.l0.addWidget(parent.min_dist_label, istretch+1, 0, 1, 1)
        parent.l0.addWidget(parent.min_dist_value, istretch+1, 1, 1, 1)
        parent.l0.addWidget(parent.n_neighbors_label, istretch+2, 0, 1, 1)
        parent.l0.addWidget(parent.n_neighbors_value, istretch+2, 1, 1, 1)
        parent.l0.addWidget(parent.n_components_label, istretch+3, 0, 1, 1)
        parent.l0.addWidget(parent.n_components_value, istretch+3, 1, 1, 1)
        parent.l0.addWidget(parent.load_umap_embedding_button, istretch+4, 0, 1, 2)
        parent.l0.addWidget(parent.cluster_method_label, istretch+5, 0, 1, 2)
        parent.l0.addWidget(parent.loadlabels_radiobutton, istretch+6, 0, 1, 1)
        parent.l0.addWidget(parent.load_cluster_labels_button, istretch+6, 1, 1, 1)
        parent.l0.addWidget(parent.kmeans_radiobutton, istretch+7, 0, 1, 1)
        parent.l0.addWidget(parent.hdbscan_radiobutton, istretch+7, 1, 1, 1)
        parent.l0.addWidget(parent.min_cluster_size_label, istretch+8, 0, 1, 1)
        parent.l0.addWidget(parent.min_cluster_size, istretch+8, 1, 1, 1)
        parent.l0.addWidget(parent.num_clusters_label, istretch+8, 0, 1, 1)
        parent.l0.addWidget(parent.num_clusters, istretch+8, 1, 1, 1)

        self.hide_umap_param(parent)
        parent.load_umap_embedding_button.clicked.connect(lambda: self.load_umap(parent))
        parent.loadlabels_radiobutton.toggled.connect(lambda: self.show_cluster_method_param(parent))
        parent.kmeans_radiobutton.toggled.connect(lambda: self.show_cluster_method_param(parent))
        parent.hdbscan_radiobutton.toggled.connect(lambda: self.show_cluster_method_param(parent))
        parent.load_cluster_labels_button.clicked.connect(lambda: self.load_cluster_labels(parent))

    def load_umap(self, parent):
        self.embedded_output = io.load_umap(parent)
        self.plot_clustering_output(parent)

    def load_cluster_labels(self, parent):
        try:
            self.ClusteringPlot_legend.clear()
        except Exception as e:
            pass
        io.load_cluster_labels(parent)
        self.plot_clustering_output(parent)

    def enable_data_clustering_features(self, parent):
        parent.data_clustering_combobox.clear()
        parent.ClusteringPlot.clear()
        # Add data to be used for clustering
        if parent.processed:
            parent.data_clustering_combobox.addItem("-- Data --")
            data_types = ["motion SVD"]#, "Running", "Pupil", "Blink"]                  # Currently for fullSVD only
            data = [parent.motSVDs[0]]#, parent.running, parent.pupil, parent.blink]    # Add ROI options
            for i in range(len(data_types)):
                if len(data[i]) > 0:
                    parent.data_clustering_combobox.addItem(data_types[i])
            parent.data_clustering_combobox.setCurrentIndex(0)
            parent.data_clustering_combobox.show()
 
            parent.run_clustering_button.show()

        cluster_method = parent.clusteringVisComboBox.currentText() ######
        if cluster_method == "UMAP":
            #parent.data_clustering_combobox.show()
            self.show_umap_param(parent)
        else:
            self.disable_data_clustering_features(parent)
        
        if self.embedded_output is not None: #and parent.clusteringVisComboBox.currentText()==self.cluster_method:
            self.show_processed_data(parent)

    def show_processed_data(self, parent):
        index = parent.data_clustering_combobox.findText(self.data_type, QtCore.Qt.MatchFixedString)
        if index >= 0:
            parent.data_clustering_combobox.setCurrentIndex(index)
        if self.cluster_labels_method == "KMeans":
            parent.kmeans_radiobutton.setChecked(True)
        elif self.cluster_labels_method == "HDBSCAN":
            parent.hdbscan_radiobutton.setChecked(True)
        elif self.cluster_labels_method == "User labels":
            parent.loadlabels_radiobutton.setChecked(True)
        else:
            parent.RadioGroup.setExclusive(False)
            parent.kmeans_radiobutton.setChecked(False)
            parent.hdbscan_radiobutton.setChecked(False)
            parent.loadlabels_radiobutton.setChecked(False)
            parent.RadioGroup.setExclusive(True)
        self.plot_clustering_output(parent)

    def disable_data_clustering_features(self, parent):
        parent.data_clustering_combobox.hide()
        parent.ClusteringPlot.clear()
        parent.zoom_in_button.hide()
        parent.zoom_out_button.hide()
        self.hide_umap_param(parent)
        parent.run_clustering_button.hide()
        parent.save_clustering_button.hide()

    def show_umap_param(self, parent):
        parent.ClusteringLabel.show()
        parent.min_dist_label.show()
        parent.min_dist_value.show()
        parent.n_neighbors_label.show()
        parent.n_neighbors_value.show()
        parent.n_components_label.show()
        parent.n_components_value.show()
        parent.cluster_method_label.show()
        parent.loadlabels_radiobutton.show()
        parent.kmeans_radiobutton.show()
        parent.hdbscan_radiobutton.show()
        self.show_cluster_method_param(parent)
        parent.load_umap_embedding_button.show()
        parent.zoom_in_button.show()
        parent.zoom_out_button.show()

    def hide_umap_param(self, parent):
        parent.ClusteringLabel.hide()
        parent.min_dist_label.hide()
        parent.min_dist_value.hide()
        parent.n_neighbors_label.hide()
        parent.n_neighbors_value.hide()
        parent.n_components_label.hide()
        parent.n_components_value.hide()
        parent.cluster_method_label.hide()
        parent.load_cluster_labels_button.hide()
        parent.loadlabels_radiobutton.hide()
        parent.kmeans_radiobutton.hide()
        parent.hdbscan_radiobutton.hide()
        parent.num_clusters_label.hide()
        parent.num_clusters.hide()
        parent.min_cluster_size_label.hide()
        parent.min_cluster_size.hide()
        parent.load_umap_embedding_button.hide()

    def show_cluster_method_param(self, parent):
        if parent.loadlabels_radiobutton.isChecked():
            parent.min_cluster_size_label.hide()
            parent.min_cluster_size.hide()
            parent.num_clusters_label.hide()
            parent.num_clusters.hide()
            parent.load_cluster_labels_button.show()
        elif parent.kmeans_radiobutton.isChecked():
            parent.min_cluster_size_label.hide()
            parent.min_cluster_size.hide()
            parent.load_cluster_labels_button.hide()
            parent.num_clusters_label.show()
            parent.num_clusters.show()
        elif parent.hdbscan_radiobutton.isChecked():
            parent.num_clusters_label.hide()
            parent.num_clusters.hide()
            parent.load_cluster_labels_button.hide()
            parent.min_cluster_size_label.show()
            parent.min_cluster_size.show()
        else:
            return

    def get_cluster_labels(self, data, parent):
        try:
            if parent.kmeans_radiobutton.isChecked():
                self.cluster_labels_method = "KMeans"
                num_clusters = int(parent.num_clusters.text())
                kmeans = MiniBatchKMeans(n_clusters=num_clusters, tol=1e-3, 
                                batch_size=100, max_iter=50)
                kmeans.fit(data)
                self.cluster_labels = kmeans.labels_
            elif parent.hdbscan_radiobutton.isChecked():
                self.cluster_labels_method = "HDBSCAN"
                clusterer = hdbscan.HDBSCAN(min_cluster_size=int(parent.min_cluster_size.text())).fit(data)
                self.cluster_labels = clusterer.labels_
            elif parent.loadlabels_radiobutton.isChecked():
                if parent.is_cluster_labels_loaded:
                    self.cluster_labels = parent.loaded_cluster_labels
                    self.cluster_labels_method = "User labels"
                else:
                    QMessageBox.about(parent, 'Error','Please load cluster labels file')
                    pass
            else:
                return
        except Exception as e:
            QMessageBox.about(parent, 'Error','Invalid input entered')
            print(e)
            pass

    def get_colors(self):
        """
        num_classes = len(np.unique(self.cluster_labels))
        colors = cm.get_cmap('gist_rainbow')(np.linspace(0, 1., num_classes))
        colors *= 255
        colors = colors.astype(int)
        colors[:,-1] = 200#127
        brushes = [pg.mkBrush(color=c) for c in colors]
        """
        num_classes = len(np.unique(self.cluster_labels))
        brushes = [pg.mkBrush(color=c) for c in colors_list[:num_classes]]
        colors = colors_list[:num_classes]
        #if -1 in np.unique(self.cluster_labels):
        #    brushes[0] = pg.mkBrush(color=(220,220,220))
        return brushes, colors

    def reset(self, parent):
        self.cluster_labels = None
        self.embedded_output = None
        self.cluster_method = None
        self.cluster_labels_method = None
        self.data_type = None
        self.n_neighbors = None
        self.min_dist = None
        self.n_components = None
        parent.clear_visualization_window()

    def set_variables(self, parent):
        try:
            self.n_neighbors = int(parent.n_neighbors_value.text())
            self.min_dist = float(parent.min_dist_value.text())
            self.n_components = int(parent.n_components_value.value())
            self.data_type = parent.data_clustering_combobox.currentText()
            self.cluster_method = parent.clusteringVisComboBox.currentText()
        except Exception as e:
            QMessageBox.about(parent, 'Error','Parameter input can only be a number')
            print(e)
            pass

    def run(self, clicked, parent):
        self.reset(parent)
        self.set_variables(parent)
        if self.data_type == "motion SVD":
            data = parent.motSVDs[0]       # Shape: num frames x num comps
            """
            elif self.data_type == "Pupil":
                data = parent.pupil
            elif self.data_type == "Blink":
                data = parent.blink
            elif self.data_type == "Running":
                data = parent.running
            """
        else:
            self.data_type = None
            msg = QMessageBox(parent)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please select data for clustering")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        if self.cluster_method == "UMAP":
            parent.update_status_bar("Data embedding using "+str(self.cluster_method))
            self.umap_embedding(data, parent)
            self.plot_clustering_output(parent)
            parent.update_status_bar("Clustering done!")
        else:
            self.hide_umap_param(parent)

    def umap_embedding(self, data, parent):
        """
        This function uses UMAP to embed loaded/processed SVD output
        """
        num_feat = data.shape[0]     # number of frames
        num_comp = data.shape[1]     # number of PCs usually 500
        self.embedded_output = umap.UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist,
                        n_components=self.n_components).fit_transform(data) # cluster features/frames

    def plot_clustering_output(self, parent):
        parent.ClusteringPlot.clear()
        num_feat = self.embedded_output.shape[0]
        num_comps = self.embedded_output.shape[1]
        is_cluster_colored = False
        all_spots_colors = [pg.mkBrush(color='w') for i in range(num_feat)]
        name = None
        
        # Get cluster labels if clustering method selected for embedded output
        if parent.kmeans_radiobutton.isChecked() or parent.hdbscan_radiobutton.isChecked() or parent.loadlabels_radiobutton.isChecked():
            self.get_cluster_labels(self.embedded_output, parent)
            brushes, colors = self.get_colors()
            name = self.cluster_labels
            if len(brushes) > 1:
                is_cluster_colored = True

        # Plot output (i) w/ cluster labels (ii) w/o  cluster labels and (iii) 3D output
        if num_comps == 2:
            # Set pixel size of embedded points on plot
            if self.embedded_output.shape[0]<500:
                point_size=6 
            elif self.embedded_output.shape[0]<2000:
                point_size=4 
            else:
                point_size=2
            if is_cluster_colored:
                scatter_plots = []
                if len(np.unique(self.cluster_labels)) > 9: #Adjust legend
                    legend_num_col = 9
                    legend_num_row = int(len(np.unique(self.cluster_labels))/legend_num_col)+1
                else:
                    legend_num_col, legend_num_row = [len(np.unique(self.cluster_labels)), 1]
                parent.ClusteringPlot_legend = pg.LegendItem(labelTextSize='11pt', horSpacing=10, 
                                                            colCount=legend_num_col, rowCount=legend_num_row)
                parent.ClusteringPlot_legend.setPos(0,20)
                for i, cluster in enumerate(np.unique(self.cluster_labels)):#range(max(self.cluster_labels)+1):
                    ind = np.where(self.cluster_labels==cluster)[0]
                    data = self.embedded_output[ind,:]
                    if cluster == -1:
                        scatter_plots.append(pg.ScatterPlotItem(data[:,0], data[:,1], symbol='o', brush=pg.mkBrush(color=(0,1,1,1)),
                                            hoverable=True, hoverSize=15, hoverSymbol="x", hoverBrush='r',
                                            pen=(0,.0001,0,0), data=ind, name=str(i)),size=point_size) #pg.mkPen(pg.hsvColor(hue=.01,sat=.01,alpha=0.01))
                    else:
                        scatter_plots.append(pg.ScatterPlotItem(data[:,0], data[:,1], symbol='o', brush=brushes[i],
                                            hoverable=True, hoverSize=15, hoverSymbol="x", hoverBrush='r',
                                            data=ind, name=str(i), size=point_size))
                    parent.ClusteringPlot.addItem(scatter_plots[i])
                    parent.ClusteringPlot_legend.addItem(scatter_plots[i], name=str(i))
                # Add all points (transparent) to connect them to hovered function
                parent.clustering_scatterplot.setData(self.embedded_output[:,0], self.embedded_output[:,1], symbol='o',
                                                 brush=(0,0,0,0),pxMode=True, hoverable=True, hoverSize=15,
                                                  hoverSymbol="x", hoverBrush='r',pen=(0,0,0,0),
                                                   data=np.arange(num_feat), name=name,size=point_size)
                parent.ClusteringPlot.addItem(parent.clustering_scatterplot)
                parent.ClusteringPlot.addItem(parent.clustering_highlight_scatterplot)
                parent.ClusteringPlot_legend.setPos(parent.clustering_scatterplot.x()+5,parent.clustering_scatterplot.y())
                parent.ClusteringPlot_legend.setParentItem(parent.ClusteringPlot)
                parent.plot_cluster_labels_p1(self.cluster_labels, colors)
            else:
                parent.clustering_scatterplot.setData(self.embedded_output[:,0], self.embedded_output[:,1], symbol='o',
                                                     brush=all_spots_colors,pxMode=True,hoverable=True, 
                                                     hoverSize=15, hoverSymbol="x",hoverBrush='r',
                                                     data=np.arange(num_feat),name=name, size=point_size)
                parent.ClusteringPlot.addItem(parent.clustering_scatterplot)
                parent.ClusteringPlot.addItem(parent.clustering_highlight_scatterplot)
            parent.ClusteringPlot.showAxis('left')
            parent.ClusteringPlot.showAxis('bottom')
            parent.ClusteringPlot.setLabels(bottom='UMAP coordinate 1',left='UMAP coordinate 2') 
        else: # 3D embedded visualization
            view = gl.GLViewWidget()
            view.setWindowTitle("3D plot of embedded points")
            plot = gl.GLScatterPlotItem()
            plot.setData(pos=self.embedded_output)
            axis = gl.GLAxisItem()
            axis.setSize(x= max(self.embedded_output[:,0]),
                        y= max(self.embedded_output[:,1]),z= max(self.embedded_output[:,2]))
            view.addItem(plot)
            view.addItem(axis)
            view.show()
        parent.save_clustering_button.show()

    def embedded_points_hovered(self, obj, ev, parent):
        """
        point_hovered = np.where(parent.clustering_scatterplot.data['hovered'])[0]
        if point_hovered.shape[0] >= 1:         # Show tooltip only when hovering over a point i.e. no empty array
            points = parent.clustering_scatterplot.points()
            vb = parent.clustering_scatterplot.getViewBox()
            if vb is not None and parent.clustering_scatterplot.opts['tip'] is not None:
                cutoff = 1                      # Display info of only one point when hovering over multiple points
                tip = [parent.clustering_scatterplot.opts['tip'](data = points[pt].data(),x=points[pt].pos().x(), y=points[pt].pos().y())
                        for pt in point_hovered[:cutoff]]
                if len(point_hovered) > cutoff:
                    tip.append('({} other...)'.format(len(point_hovered) - cutoff))
                vb.setToolTip('\n\n'.join(tip))
                frame = str(points[point_hovered[0]].data())#np.random.randint(len(point_hovered))]].data())
                parent.setFrame.setText(frame)  
        """
        if len(ev) > 0:
            new = parent.clustering_scatterplot._maskAt(ev[0].pos())
            points = parent.clustering_scatterplot.points()[new][::-1]#              # display frame from one of the hovered points
            # Show information about hovered points in a tool tip
            vb = parent.clustering_scatterplot.getViewBox()
            if vb is not None and parent.clustering_scatterplot.opts['tip'] is not None:
                cutoff = 2
                tip = [parent.clustering_scatterplot.opts['tip'](x=pt.pos().x(), y=pt.pos().y(), data=pt.data())
                        for pt in points[:cutoff]]
                if len(points) > cutoff:
                    tip.append('({} others...)'.format(len(points) - cutoff))
                vb.setToolTip('\n\n'.join(tip))
                frame = str(points[np.random.randint(len(points))].data())#np.random.randint(len(point_hovered))]].data())
                parent.setFrame.setText(frame)

    def mouse_moved_embedding(self, pos, parent):
        if self.embedded_output is not None:
            if parent.ClusteringPlot.sceneBoundingRect().contains(pos):
                x = parent.ClusteringPlot.vb.mapSceneToView(pos).x()
                y = parent.ClusteringPlot.vb.mapSceneToView(pos).y()
                scatter_x = np.array([i.pos().x() for i in parent.clustering_scatterplot.points()])
                scatter_y = np.array([i.pos().y() for i in parent.clustering_scatterplot.points()])
                dists = (scatter_x - x)**2 + (scatter_y - y)**2
                data = np.argmin(dists.flatten()).astype(int)
                #data = parent.clustering_scatterplot.points()[parent.clustering_scatterplot._maskAt(pos)]
                print(data)
                parent.setFrame.setText(str(data))

    def save_dialog(self, clicked, parent):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Save as")
        dialog.verticalLayout = QtWidgets.QVBoxLayout(dialog)

        dialog.label = QtWidgets.QLabel(dialog)
        dialog.label.setAlignment(QtCore.Qt.AlignCenter)
        dialog.label.setTextFormat(QtCore.Qt.RichText)
        dialog.label.setText("Save files:")
        dialog.verticalLayout.addWidget(dialog.label)

        dialog.data_checkbox = QCheckBox("Cluster data (*.npy)")
        dialog.data_checkbox.setChecked(True)
        dialog.videos_checkbox = QCheckBox("Cluster videos (*.avi)")
        dialog.videos_checkbox.stateChanged.connect(lambda: self.enable_video_options(dialog))

        dialog.num_frames_label = QLabel("#Frames/cluster:")
        dialog.num_frames = QLineEdit()
        dialog.num_frames.setText(str(30))
        dialog.num_frames.setEnabled(False)
        dialog.fps_label = QLabel("FPS:")
        dialog.fps = QLineEdit()
        dialog.fps.setText(str(10.0))
        dialog.fps.setEnabled(False)

        dialog.ok_button = QPushButton('Ok')
        dialog.ok_button.setDefault(True)
        dialog.ok_button.clicked.connect(lambda: self.ok_save(dialog, parent))
        dialog.cancel_button = QPushButton('Cancel')
        dialog.cancel_button.clicked.connect(dialog.close)

        # Add options to dialog box
        dialog.verticalLayout.addWidget(dialog.data_checkbox)
        dialog.verticalLayout.addWidget(dialog.videos_checkbox)
        dialog.widget = QtWidgets.QWidget(dialog)
        dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget)
        dialog.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        dialog.horizontalLayout.setObjectName("horizontalLayout")
        dialog.verticalLayout.addWidget(dialog.widget)
        dialog.horizontalLayout.addWidget(dialog.num_frames_label)
        dialog.horizontalLayout.addWidget(dialog.num_frames)
        dialog.horizontalLayout.addWidget(dialog.fps_label)
        dialog.horizontalLayout.addWidget(dialog.fps)
        dialog.widget2 = QtWidgets.QWidget(dialog)
        dialog.horizontalLayout = QtWidgets.QHBoxLayout(dialog.widget2)
        dialog.horizontalLayout.addWidget(dialog.cancel_button)
        dialog.horizontalLayout.addWidget(dialog.ok_button)
        dialog.verticalLayout.addWidget(dialog.widget2)

        dialog.adjustSize()
        dialog.exec_()

    def enable_video_options(self, dialogBox):
        if dialogBox.videos_checkbox.isChecked():
            dialogBox.fps.setEnabled(True)
            dialogBox.num_frames.setEnabled(True)
        else:
            dialogBox.fps.setEnabled(False)
            dialogBox.num_frames.setEnabled(False)

    def ok_save(self, dialogBox, parent):
        if dialogBox.videos_checkbox.isChecked():
            if len(np.unique(self.cluster_labels)) > 1:
                try:
                    self.save_cluster_video(parent, float(dialogBox.fps.text()), int(dialogBox.num_frames.text()))
                except Exception as e:
                    QMessageBox.about(parent, 'Error','Invalid input entered')
                    print(e)
                    pass
            else:
                msg = QMessageBox(parent)
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Please generate cluster labels for saving cluster videos")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
        if dialogBox.data_checkbox.isChecked():
            self.save_cluster_output(parent)
        # Done!
        dialogBox.close()

    def save_cluster_output(self, parent):
        #Check if len(parent.filenames)==1 to enable clustering and gifs
        output = {"data_name": self.data_type,
                "cluster_method": self.cluster_method,
                "n_neighbors": self.n_neighbors,
                "min_dist": self.min_dist,
                "n_components": self.n_components,
                "embedded_output": self.embedded_output,
                "cluster_labels_method": self.cluster_labels_method,
                "cluster_labels": self.cluster_labels}
        io.save_clustering_output(output, parent)

    def save_cluster_video(self, parent, fps, num_frames_gif):
        """
        Write frames from each cluster to a different video (*.avi)
        """
        cumframes, Ly, Lx, capture = utils.get_frame_details(parent.filenames)
        capture = capture[0][0]
        num_clusters = len(np.unique(self.cluster_labels))
        filename, ext = parent.filenames[0][0].split(".")
        filename = filename.split("/")[-1]    # Use video filename 
        
        # Create 2D list of random frames selected from each cluster
        cluster_frames_2D_list = np.zeros((num_clusters, num_frames_gif))
        for c, clusterid in enumerate(np.unique(self.cluster_labels)):  
            clusterids = np.where(clusterid == self.cluster_labels)[0]
            numimagesofcluster = len(clusterids)
            for i in range(num_frames_gif):
                if numimagesofcluster > 0:
                    cluster_frames_2D_list[c][i] = clusterids[np.random.randint(numimagesofcluster)]
        
        # Create video of frames selected from each cluster
        for cluster, cluster_frame_list in enumerate(cluster_frames_2D_list):
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            ## Improve video resolution
            savename = os.path.join(parent.save_path, ("{f}_cluster{c}.avi".format(f=filename, c=cluster)))
            video_writer = cv2.VideoWriter(savename,fourcc=fourcc, fps=fps, frameSize=(Lx[0], Ly[0]))
            for j, frame_ind in enumerate(cluster_frame_list):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
                ret = True
                ret, frame = capture.read()
                if ret:
                    video_writer.write(frame)
                else:
                    print('img load failed, breaking')
