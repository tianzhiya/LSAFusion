import torch
import torch.nn as nn

# 假设你有一个输入 Tensor input_tensor，大小为 [1, 1, 256, 256]
input_tensor = torch.randn(1, 1, 256, 256)


# 创建一个包含卷积模块和插值操作的模型，将输入从 [1, 1, 256, 256] 上采样为 [1, 48, 640, 480]
class UpsampleModel(nn.Module):
    def __init__(self, height, width):
        super(UpsampleModel, self).__init__()
        self.conv = nn.Conv2d(1, 48, kernel_size=3, padding=1)  # 卷积层
        self.upsample = nn.Upsample(size=(height, width), mode='bilinear', align_corners=False)  # 插值操作

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


model = UpsampleModel(640, 480)

output_tensor = model(input_tensor)

# 输出大小为 [1, 48, 640, 480]
print(output_tensor.size())
