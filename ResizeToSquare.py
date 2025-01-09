import torch
import torch.nn as nn
import torch.nn.functional as F


class ResizeToSquare(nn.Module):
    def __init__(self, target_size):
        super(ResizeToSquare, self).__init__()
        self.target_size = target_size

    def forward(self, input_tensor):
        _, _, height, width = input_tensor.size()

        # Calculate the aspect ratio of the input image
        aspect_ratio = width / height

        if aspect_ratio > 1:
            # Landscape orientation (wider than tall)
            new_width = self.target_size
            new_height = self.target_size
        else:
            # Portrait orientation (taller than wide)
            new_width = self.target_size
            new_height = self.target_size

        # Resize the input tensor to the new dimensions using bilinear interpolation
        resized_tensor = F.interpolate(input_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)

        return resized_tensor


# Example usage
resize_module = ResizeToSquare(target_size=256)
image = torch.randn(1, 3, 281, 375)  # Replace with your actual image tensor
output_image = resize_module(image)
print(output_image.size())  # This should print torch.Size([1, 3, 300, 300])
