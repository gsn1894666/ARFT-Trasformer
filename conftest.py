import pytest
import torch
from ResNet import DANNet  # 假设你的模型类在models.py中

@pytest.fixture
def model():
    # 创建一个模型实例
    model = DANNet(num_classes=31)
    # 如果使用 GPU，放到 CUDA 上
    if torch.cuda.is_available():
        model.cuda()
    return model
