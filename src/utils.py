from torchvision import transforms
from PIL import Image
import torch


def preprocess_image(image, size=512):
    """
    Convert PIL image to PyTorch tensor
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)

    return image


def tensor_to_image(tensor):
    """
    Convert PyTorch tensor to PIL image
    """
    image = tensor.detach().cpu()

    # Remove batch dimension
    image = image.squeeze(0)

    # Keep pixel values between 0 and 1
    image = image.clamp(0, 1)

    # Convert CHW -> HWC
    image = image.permute(1, 2, 0)

    # Convert to numpy image
    image = (image.numpy() * 255).astype("uint8")

    return Image.fromarray(image)
