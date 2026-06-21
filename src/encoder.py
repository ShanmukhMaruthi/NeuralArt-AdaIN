import torch
import torch.nn as nn


class VGGEncoder(nn.Module):
    def __init__(self, vgg_path):
        super(VGGEncoder, self).__init__()

        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),

        )

        self.vgg.load_state_dict(torch.load(vgg_path))

        self.enc_1 = nn.Sequential(*list(self.vgg.children())[:4])
        self.enc_2 = nn.Sequential(*list(self.vgg.children())[4:11])
        self.enc_3 = nn.Sequential(*list(self.vgg.children())[11:18])
        self.enc_4 = nn.Sequential(*list(self.vgg.children())[18:31])

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def forward(self, x):
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)

        return h4
