from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1  # 只有一张图片
    
    transform = T.Compose([
        T.ToTensor(),  # 转为 Tensor
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])

    def __getitem__(self, idx):
        # 加载图片
        image = Image.open(self.image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_path.split('/')[-1]  # 返回图片和文件名