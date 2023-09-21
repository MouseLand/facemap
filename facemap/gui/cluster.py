"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os

import cv2
import numpy as np
import pyqtgraph as pg

try:
    import umap

    UMAP_INSTALLED = True
except:
    UMAP_INSTALLED = False

# todo: use opentsne for embedding
TSNE_INSTALLED = True

# import hdbscan
from matplotlib import cm
from qtpy import QtCore, QtWidgets
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
)

# import pyqtgraph.opengl as gl
from sklearn.cluster import MiniBatchKMeans

from facemap import utils
from facemap.gui import io


class Cluster:
    def __init__(
        self,
        parent,
        method=None,
        cluster_labels=None,
        cluster_labels_method=None,
        data_type=None,
    ):
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
        parent.clustering_label = QLabel("Clustering")
        parent.clustering_label.setStyleSheet("color: white;")
        parent.clustering_label.setAlignment(QtCore.Qt.AlignCenter)
        parent.clustering_label.setFont(QFont("Arial", 12, QFont.Bold))

        parent.load_embedding_button = QPushButton("Load embedding")
        parent.load_embedding_button.clicked.connect(lambda: self.load_umap(parent))
        parent.load_cluster_labels_button = QPushButton("Load labels")
        parent.load_cluster_labels_button.clicked.connect(
            lambda: self.load_cluster_labels(parent)
        )
        """
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
        # metric
        parent.cluster_method_label = QLabel("Cluster labels")
        parent.cluster_method_label.setStyleSheet("color: gray;")
        parent.cluster_method_label.setAlignment(QtCore.Qt.AlignCenter)
        parent.RadioGroup = QButtonGroup()
        parent.load_cluster_labels_button = QPushButton("Load")
        parent.loadlabels_radiobutton = QRadioButton("Load labels")
        parent.loadlabels_radiobutton.setStyleSheet("color: gray;")
        parent.RadioGroup.addButton(parent.loadlabels_radiobutton)
        parent.kmeans_radiobutton = QRadioButton("KMeans")
        parent.kmeans_radiobutton.setStyleSheet("color: gray;")
        parent.RadioGroup.addButton(parent.kmeans_radiobutton)
        # parent.hdbscan_radiobutton = QRadioButton("HDBSCAN")
        # parent.hdbscan_radiobutton.setStyleSheet("color: gray;")
        # parent.RadioGroup.addButton(parent.hdbscan_radiobutton)
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
        parent.scene_grid_layout.addWidget(parent.clustering_label, istretch, 0, 1, 2)
        parent.scene_grid_layout.addWidget(parent.min_dist_label, istretch + 1, 0, 1, 1)
        parent.scene_grid_layout.addWidget(parent.min_dist_value, istretch + 1, 1, 1, 1)
        parent.scene_grid_layout.addWidget(
            parent.n_neighbors_label, istretch + 2, 0, 1, 1
        )
        parent.scene_grid_layout.addWidget(
            parent.n_neighbors_value, istretch + 2, 1, 1, 1
        )
        parent.scene_grid_layout.addWidget(
            parent.n_components_label, istretch + 3, 0, 1, 1
        )
        parent.scene_grid_layout.addWidget(
            parent.n_components_value, istretch + 3, 1, 1, 1
        )
        parent.scene_grid_layout.addWidget(
            parent.load_embedding_button, istretch + 4, 0, 1, 2
        )
        parent.scene_grid_layout.addWidget(
            parent.cluster_method_label, istretch + 5, 0, 1, 2
        )
        parent.scene_grid_layout.addWidget(
            parent.loadlabels_radiobutton, istretch + 6, 0, 1, 1
        )
        parent.scene_grid_layout.addWidget(
            parent.load_cluster_labels_button, istretch + 6, 1, 1, 1
        )
        parent.scene_grid_layout.addWidget(
            parent.kmeans_radiobutton, istretch + 7, 0, 1, 1
        )
        # parent.scene_grid_layout.addWidget(parent.hdbscan_radiobutton, istretch+7, 1, 1, 1)
        parent.scene_grid_layout.addWidget(
            parent.min_cluster_size_label, istretch + 8, 0, 1, 1
        )
        parent.scene_grid_layout.addWidget(
            parent.min_cluster_size, istretch + 8, 1, 1, 1
        )
        parent.scene_grid_layout.addWidget(
            parent.num_clusters_label, istretch + 8, 0, 1, 1
        )
        parent.scene_grid_layout.addWidget(parent.num_clusters, istretch + 8, 1, 1, 1)
        self.hide_umap_param(parent)
        parent.load_embedding_button.clicked.connect(lambda: self.load_umap(parent))
        parent.loadlabels_radiobutton.toggled.connect(
            lambda: self.show_cluster_method_param(parent)
        )
        parent.kmeans_radiobutton.toggled.connect(
            lambda: self.show_cluster_method_param(parent)
        )
        # parent.hdbscan_radiobutton.toggled.connect(lambda: self.show_cluster_method_param(parent))
        parent.load_cluster_labels_button.clicked.connect(
            lambda: self.load_cluster_labels(parent)
        )
        """
        istretch = 13
        parent.scene_grid_layout.addWidget(parent.clustering_label, istretch, 0, 1, 2)
        parent.scene_grid_layout.addWidget(
            parent.load_embedding_button, istretch + 1, 0, 1, 1
        )
        parent.scene_grid_layout.addWidget(
            parent.load_cluster_labels_button, istretch + 1, 1, 1, 1
        )
        self.hide_umap_param(parent)

    def load_umap(self, parent):
        self.embedded_output = io.load_umap(parent)
        self.plot_clustering_output(parent)

    def load_cluster_labels(self, parent):
        try:
            self.clustering_plot_legend.clear()
        except Exception as e:
            pass
        io.load_cluster_labels(parent)
        self.plot_clustering_output(parent)

    def enable_data_clustering_features(self, parent):
        parent.data_clustering_combobox.clear()
        parent.clustering_plot.clear()
        # Add data to be used for clustering
        if parent.processed:
            parent.data_clustering_combobox.addItem("-- Data --")
            data_types = [
                "motion SVD"
            ]  # , "Running", "Pupil", "Blink"]                  # Currently for fullSVD only
            data = [
                parent.motSVDs[0]
            ]  # , parent.running, parent.pupil, parent.blink]    # Add ROI options
            for i in range(len(data_types)):
                if len(data[i]) > 0:
                    parent.data_clustering_combobox.addItem(data_types[i])
            parent.data_clustering_combobox.setCurrentIndex(0)
            # parent.data_clustering_combobox.show()

            parent.run_clustering_button.show()

        embed_method = parent.roi_embed_combobox.currentText()  ######
        if embed_method == "UMAP" and UMAP_INSTALLED:
            self.show_umap_param(parent)
        elif embed_method == "tSNE" and TSNE_INSTALLED:
            self.hide_umap_param(parent)
            self.show_tsne_options(parent)
        else:
            self.disable_data_clustering_features(parent)

        if (
            self.embedded_output is not None
        ):  # and parent.roi_embed_combobox.currentText()==self.cluster_method:
            self.show_processed_data(parent)

    def show_processed_data(self, parent):
        index = parent.data_clustering_combobox.findText(
            self.data_type, QtCore.Qt.MatchFixedString
        )
        if index >= 0:
            parent.data_clustering_combobox.setCurrentIndex(index)
        if self.cluster_labels_method == "KMeans":
            parent.kmeans_radiobutton.setChecked(True)
        # elif self.cluster_labels_method == "HDBSCAN":
        #    parent.hdbscan_radiobutton.setChecked(True)
        elif self.cluster_labels_method == "User labels":
            parent.loadlabels_radiobutton.setChecked(True)
        else:
            parent.RadioGroup.setExclusive(False)
            parent.kmeans_radiobutton.setChecked(False)
            # parent.hdbscan_radiobutton.setChecked(False)
            parent.loadlabels_radiobutton.setChecked(False)
            parent.RadioGroup.setExclusive(True)
        self.plot_clustering_output(parent)

    def disable_data_clustering_features(self, parent):
        parent.data_clustering_combobox.hide()
        parent.clustering_plot.clear()
        parent.zoom_in_button.hide()
        parent.zoom_out_button.hide()
        self.hide_umap_param(parent)
        parent.run_clustering_button.hide()
        parent.save_clustering_button.hide()

    def show_umap_param(self, parent):
        """
        parent.min_dist_label.show()
        parent.min_dist_value.show()
        parent.n_neighbors_label.show()
        parent.n_neighbors_value.show()
        parent.n_components_label.show()
        parent.n_components_value.show()
        parent.cluster_method_label.show()
        parent.loadlabels_radiobutton.show()
        parent.kmeans_radiobutton.show()
        # parent.hdbscan_radiobutton.show()
        self.show_cluster_method_param(parent)
        """
        parent.clustering_label.show()
        parent.load_embedding_button.show()
        parent.zoom_in_button.show()
        parent.zoom_out_button.show()
        parent.load_cluster_labels_button.show()

    def hide_umap_param(self, parent):
        """
        parent.min_dist_label.hide()
        parent.min_dist_value.hide()
        parent.n_neighbors_label.hide()
        parent.n_neighbors_value.hide()
        parent.n_components_label.hide()
        parent.n_components_value.hide()
        parent.cluster_method_label.hide()
        parent.loadlabels_radiobutton.hide()
        parent.kmeans_radiobutton.hide()
        # parent.hdbscan_radiobutton.hide()
        parent.num_clusters_label.hide()
        parent.num_clusters.hide()
        parent.min_cluster_size_label.hide()
        parent.min_cluster_size.hide()
        """
        parent.clustering_label.hide()
        parent.load_embedding_button.hide()
        parent.load_cluster_labels_button.hide()

    def show_tsne_options(self, parent):
        parent.clustering_label.show()
        parent.load_embedding_button.show()
        parent.load_cluster_labels_button.show()

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
        else:
            return
        """
        elif parent.hdbscan_radiobutton.isChecked():
            parent.num_clusters_label.hide()
            parent.num_clusters.hide()
            parent.load_cluster_labels_button.hide()
            parent.min_cluster_size_label.show()
            parent.min_cluster_size.show()
         """

    def get_cluster_labels(self, data, parent):
        if parent.is_cluster_labels_loaded:
            self.cluster_labels = parent.loaded_cluster_labels
            self.cluster_labels_method = "User labels"
        """
        try:
            if parent.kmeans_radiobutton.isChecked():
                self.cluster_labels_method = "KMeans"
                num_clusters = int(parent.num_clusters.text())
                kmeans = MiniBatchKMeans(
                    n_clusters=num_clusters, tol=1e-3, batch_size=100, max_iter=50
                )
                kmeans.fit(data)
                self.cluster_labels = kmeans.labels_
            elif parent.loadlabels_radiobutton.isChecked():
                if parent.is_cluster_labels_loaded:
                    self.cluster_labels = parent.loaded_cluster_labels
                    self.cluster_labels_method = "User labels"
                else:
                    QMessageBox.about(
                        parent, "Error", "Please load cluster labels file"
                    )
                    pass
            else:
                return
        except Exception as e:
            QMessageBox.about(parent, "Error", "Invalid input entered")
            print(e)
            pass
        """

    def get_colors(self):
        num_classes = len(np.unique(self.cluster_labels))
        colors = cm.get_cmap("gist_rainbow")(np.linspace(0, 1.0, num_classes))
        colors *= 255
        colors = colors.astype(int)
        colors[:, -1] = 200  # 127
        brushes = [pg.mkBrush(color=c) for c in colors]
        """
        num_classes = len(np.unique(self.cluster_labels))
        brushes = [pg.mkBrush(color=c) for c in colors_list[:num_classes]]
        colors = colors_list[:num_classes]"""
        # if -1 in np.unique(self.cluster_labels):
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
            self.cluster_method = parent.roi_embed_combobox.currentText()
        except Exception as e:
            QMessageBox.about(parent, "Error", "Parameter input can only be a number")
            print(e)
            pass

    def run(self, clicked, parent):
        self.reset(parent)
        self.set_variables(parent)
        if self.data_type == "motion SVD":
            data = parent.motSVDs[0]  # Shape: num frames x num comps
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
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Please select data for clustering")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec_()
            return
        if self.cluster_method == "UMAP":
            parent.update_status_bar("Data embedding using " + str(self.cluster_method))
            self.umap_embedding(data, parent)
            self.plot_clustering_output(parent)
            parent.update_status_bar("Clustering done!")
        else:
            self.hide_umap_param(parent)

    def umap_embedding(self, data, parent):
        """
        This function uses UMAP to embed loaded/processed SVD output
        """
        num_feat = data.shape[0]  # number of frames
        num_comp = data.shape[1]  # number of PCs usually 500
        self.embedded_output = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
        ).fit_transform(
            data
        )  # cluster features/frames

    def plot_clustering_output(self, parent):
        parent.clustering_plot.clear()
        num_feat = self.embedded_output.shape[0]
        num_comps = self.embedded_output.shape[1]
        is_cluster_colored = False
        all_spots_colors = [pg.mkBrush(color="w") for i in range(num_feat)]
        name = None

        # Get cluster labels if labels are loaded
        self.get_cluster_labels(self.embedded_output, parent)
        brushes, colors = self.get_colors()
        name = self.cluster_labels
        if len(brushes) > 1:
            is_cluster_colored = True

        # Plot output (i) w/ cluster labels (ii) w/o  cluster labels and (iii) 3D output
        if num_comps == 2:
            # Set pixel size of embedded points on plot
            if self.embedded_output.shape[0] < 500:
                point_size = 0.007 * parent.sizeObject.height()
            elif self.embedded_output.shape[0] < 2000:
                point_size = 0.005 * parent.sizeObject.height()
            else:
                point_size = 0.003 * parent.sizeObject.height()
            if is_cluster_colored:
                scatter_plots = []
                if len(np.unique(self.cluster_labels)) > 9:  # Adjust legend
                    legend_num_col = 9
                    legend_num_row = (
                        int(len(np.unique(self.cluster_labels)) / legend_num_col) + 1
                    )
                else:
                    legend_num_col, legend_num_row = [
                        len(np.unique(self.cluster_labels)),
                        1,
                    ]
                parent.clustering_plot_legend = pg.LegendItem(
                    labelTextSize="11pt",
                    horSpacing=10,
                    colCount=legend_num_col,
                    rowCount=legend_num_row,
                )
                parent.clustering_plot_legend.setPos(0, 20)
                for i, cluster in enumerate(
                    np.unique(self.cluster_labels)
                ):  # range(max(self.cluster_labels)+1):
                    ind = np.where(self.cluster_labels == cluster)[0]
                    data = self.embedded_output[ind, :]
                    if cluster == -1:
                        scatter_plots.append(
                            pg.ScatterPlotItem(
                                data[:, 0],
                                data[:, 1],
                                symbol="o",
                                brush=pg.mkBrush(color=(0, 1, 1, 1)),
                                hoverable=True,
                                hoverSize=15,
                                hoverSymbol="x",
                                hoverBrush="r",
                                pen=(0, 0.0001, 0, 0),
                                data=ind,
                                name=str(cluster),
                            ),
                            size=point_size,
                        )  # pg.mkPen(pg.hsvColor(hue=.01,sat=.01,alpha=0.01))
                    else:
                        scatter_plots.append(
                            pg.ScatterPlotItem(
                                data[:, 0],
                                data[:, 1],
                                symbol="o",
                                brush=brushes[i],
                                hoverable=True,
                                hoverSize=15,
                                hoverSymbol="x",
                                hoverBrush="r",
                                data=ind,
                                name=str(cluster),
                                size=point_size,
                            )
                        )
                    parent.clustering_plot.addItem(scatter_plots[i])
                    parent.clustering_plot_legend.addItem(
                        scatter_plots[i], name=str(cluster)
                    )
                # Add all points (transparent) to connect them to hovered function
                parent.clustering_scatterplot.setData(
                    self.embedded_output[:, 0],
                    self.embedded_output[:, 1],
                    symbol="o",
                    brush=(0, 0, 0, 0),
                    pxMode=True,
                    hoverable=True,
                    hoverSize=15,
                    hoverSymbol="x",
                    hoverBrush="r",
                    pen=(0, 0, 0, 0),
                    data=np.arange(num_feat),
                    name=name,
                    size=point_size,
                )
                parent.clustering_plot.addItem(parent.clustering_scatterplot)
                parent.clustering_plot.addItem(parent.clustering_highlight_scatterplot)
                parent.clustering_plot_legend.setPos(
                    parent.clustering_scatterplot.x() + 5,
                    parent.clustering_scatterplot.y(),
                )
                parent.clustering_plot_legend.setParentItem(parent.clustering_plot)
                parent.plot_cluster_labels(self.cluster_labels, colors)
            else:
                parent.clustering_scatterplot.setData(
                    self.embedded_output[:, 0],
                    self.embedded_output[:, 1],
                    symbol="o",
                    brush=all_spots_colors,
                    pxMode=True,
                    hoverable=True,
                    hoverSize=15,
                    hoverSymbol="x",
                    hoverBrush="r",
                    data=np.arange(num_feat),
                    name=name,
                    size=point_size,
                )
                parent.clustering_plot.addItem(parent.clustering_scatterplot)
                parent.clustering_plot.addItem(parent.clustering_highlight_scatterplot)
            parent.clustering_plot.showAxis("left")
            parent.clustering_plot.showAxis("bottom")
            parent.clustering_plot.setLabels(bottom="Dimension 1", left="Dimension 2")
        else:
            """
            # 3D embedded visualization
            view = gl.GLViewWidget()
            view.setWindowTitle("3D plot of embedded points")
            plot = gl.GLScatterPlotItem()
            plot.setData(pos=self.embedded_output)
            axis = gl.GLAxisItem()
            axis.setSize(
                x=max(self.embedded_output[:, 0]),
                y=max(self.embedded_output[:, 1]),
                z=max(self.embedded_output[:, 2]),
            )
            view.addItem(plot)
            view.addItem(axis)
            view.show()
            """
            # Raise error that data type not supported
            raise ValueError("Data type not supported. Please use 2D data.")
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
                parent.current_frame_lineedit.setText(frame)
        """
        if len(ev) > 0:
            new = parent.clustering_scatterplot._maskAt(ev[0].pos())
            points = parent.clustering_scatterplot.points()[new][
                ::-1
            ]  #              # display frame from one of the hovered points
            # Show information about hovered points in a tool tip
            vb = parent.clustering_scatterplot.getViewBox()
            if vb is not None and parent.clustering_scatterplot.opts["tip"] is not None:
                cutoff = 2
                tip = [
                    parent.clustering_scatterplot.opts["tip"](
                        x=pt.pos().x(), y=pt.pos().y(), data=pt.data()
                    )
                    for pt in points[:cutoff]
                ]
                if len(points) > cutoff:
                    tip.append("({} others...)".format(len(points) - cutoff))
                vb.setToolTip("\n\n".join(tip))
                frame = str(
                    points[np.random.randint(len(points))].data()
                )  # np.random.randint(len(point_hovered))]].data())
                parent.current_frame_lineedit.setText(frame)

    def mouse_moved_embedding(self, pos, parent):
        if self.embedded_output is not None:
            if parent.clustering_plot.sceneBoundingRect().contains(pos):
                x = parent.clustering_plot.vb.mapSceneToView(pos).x()
                y = parent.clustering_plot.vb.mapSceneToView(pos).y()
                scatter_x = np.array(
                    [i.pos().x() for i in parent.clustering_scatterplot.points()]
                )
                scatter_y = np.array(
                    [i.pos().y() for i in parent.clustering_scatterplot.points()]
                )
                dists = (scatter_x - x) ** 2 + (scatter_y - y) ** 2
                data = np.argmin(dists.flatten()).astype(int)
                # data = parent.clustering_scatterplot.points()[parent.clustering_scatterplot._maskAt(pos)]
                print(data)
                parent.current_frame_lineedit.setText(str(data))

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
        dialog.videos_checkbox.stateChanged.connect(
            lambda: self.enable_video_options(dialog)
        )

        dialog.num_frames_label = QLabel("#Frames/cluster:")
        dialog.num_frames = QLineEdit()
        dialog.num_frames.setText(str(30))
        dialog.num_frames.setEnabled(False)
        dialog.fps_label = QLabel("FPS:")
        dialog.fps = QLineEdit()
        dialog.fps.setText(str(10.0))
        dialog.fps.setEnabled(False)

        dialog.ok_button = QPushButton("Ok")
        dialog.ok_button.setDefault(True)
        dialog.ok_button.clicked.connect(lambda: self.ok_save(dialog, parent))
        dialog.cancel_button = QPushButton("Cancel")
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
                    self.save_cluster_video(
                        parent,
                        float(dialogBox.fps.text()),
                        int(dialogBox.num_frames.text()),
                    )
                except Exception as e:
                    QMessageBox.about(parent, "Error", "Invalid input entered")
                    print(e)
                    pass
            else:
                msg = QMessageBox(parent)
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setText("Please generate cluster labels for saving cluster videos")
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec_()
        if dialogBox.data_checkbox.isChecked():
            self.save_cluster_output(parent)
        # Done!
        dialogBox.close()

    def save_cluster_output(self, parent):
        # Check if len(parent.filenames)==1 to enable clustering and gifs
        output = {
            "data_name": self.data_type,
            "cluster_method": self.cluster_method,
            "n_neighbors": self.n_neighbors,
            "min_dist": self.min_dist,
            "n_components": self.n_components,
            "embedded_output": self.embedded_output,
            "cluster_labels_method": self.cluster_labels_method,
            "cluster_labels": self.cluster_labels,
        }
        io.save_clustering_output(output, parent)

    def save_cluster_video(self, parent, fps, num_frames_gif):
        """
        Write frames from each cluster to a different video (*.avi)
        """
        cumframes, Ly, Lx, capture = utils.get_frame_details(parent.filenames)
        capture = capture[0][0]
        num_clusters = len(np.unique(self.cluster_labels))
        filename, ext = os.path.splitext(
            parent.filenames[0][0]
        )  # parent.filenames[0][0].split(".")
        filename = os.path.basename(
            filename
        )  # filename.split("/")[-1]  # Use video filename

        # Create 2D list of random frames selected from each cluster
        cluster_frames_2D_list = np.zeros((num_clusters, num_frames_gif))
        for c, clusterid in enumerate(np.unique(self.cluster_labels)):
            clusterids = np.where(clusterid == self.cluster_labels)[0]
            numimagesofcluster = len(clusterids)
            for i in range(num_frames_gif):
                if numimagesofcluster > 0:
                    cluster_frames_2D_list[c][i] = clusterids[
                        np.random.randint(numimagesofcluster)
                    ]

        # Create video of frames selected from each cluster
        for cluster, cluster_frame_list in enumerate(cluster_frames_2D_list):
            fourcc = cv2.VideoxWriter_fourcc("M", "J", "P", "G")
            ## Improve video resolution
            savename = os.path.join(
                parent.save_path, ("{f}_cluster{c}.avi".format(f=filename, c=cluster))
            )
            video_writer = cv2.VideoWriter(
                savename, fourcc=fourcc, fps=fps, frameSize=(Lx[0], Ly[0])
            )
            for j, frame_ind in enumerate(cluster_frame_list):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
                ret = True
                ret, frame = capture.read()
                if ret:
                    video_writer.write(frame)
                else:
                    print("img load failed, breaking")
