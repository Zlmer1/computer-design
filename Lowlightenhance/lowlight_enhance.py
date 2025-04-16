import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
# from net.CIDNet import CIDNet
from net.retinex import Retinex
from skimage import img_as_ubyte
from data.singledata import SingleImageDataset
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog


def load_model():
    #当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #预训练模型的路径
    weighth_path = os.path.join(current_dir, 'pretrained_models', 'lowlight_enhance.pth')

    model = Retinex().cuda()

    model.load_state_dict(torch.load(weighth_path, map_location='cuda'))
    model.eval()

    return model

def enhance_image(input_image, model):
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

    transform = SingleImageDataset.transform
    input = DataLoader(SingleImageDataset(input_image, transform=transform), batch_size=1, shuffle=False)
    # input_tensor = transform(input_image).unsqueeze(0).cuda()
    input_tensor = next(iter(input))[0].cuda()

    with torch.no_grad():
        output, _ = model(input_tensor)
        output = torch.clamp(output, 0, 1)
    output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
    return output_img

class LowLightEnhanceUI(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.initUI()

    def initUI(self):
        self.setWindowTitle('低光增强')
        self.setGeometry(100, 100, 800, 600)

        layout = QtWidgets.QVBoxLayout()

        # 原始图像显示
        self.original_image_label = QtWidgets.QLabel('原始图像')
        self.original_image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.original_image_label)

        # 增强后图像显示
        self.enhanced_image_label = QtWidgets.QLabel('增强后图像')
        self.enhanced_image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.enhanced_image_label)

        # 文件选择按钮
        self.select_button = QtWidgets.QPushButton('选择文件')
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        # 增强按钮
        self.enhance_button = QtWidgets.QPushButton('开始增强')
        self.enhance_button.clicked.connect(self.enhance_image)
        layout.addWidget(self.enhance_button)

        # 保存按钮
        self.save_button = QtWidgets.QPushButton('保存文件')
        self.save_button.clicked.connect(self.save_file)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def select_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.file_path = file_path
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.original_image_label)
            self.enhance_button.setEnabled(True)  # 启用“开始增强”按钮
    
    def enhance_image(self):
        if hasattr(self, 'original_image'):
            # Enhance the image
            self.enhanced_image = enhance_image(self.original_image, self.model)
            self.display_image(self.enhanced_image, self.enhanced_image_label)
            self.save_button.setEnabled(True)  # 启用“保存文件”按钮

    def display_image(self, image, label):
        if isinstance(image, Image.Image):  # PIL Image
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(400, 300, QtCore.Qt.KeepAspectRatio))

    def save_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "Images (*.png *.jpg)", options=options)
        if file_path:
            self.enhanced_image.save(file_path)
            QtWidgets.QMessageBox.information(self, "保存成功", "图像已保存")

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)

    model = load_model()
    low_light_enhance_ui = LowLightEnhanceUI(model)
    low_light_enhance_ui.show()

    sys.exit(app.exec_())