import torch
import torch.nn as nn


class VGGEncoder(nn.Module):
    def __init__(self, vgg_path, device="cpu"):
        super(VGGEncoder, self).__init__()

        self.vgg = nn.Sequential(
            # relu1_1
            nn.Conv2d(3, 3, 1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, 3),
            nn.ReLU(inplace=True),

            # relu1_2
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            # relu2_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),

            # relu2_2
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            # relu3_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),

            # relu3_2
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),

            # relu3_3
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),

            # relu3_4
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            # relu4_1 (last layer used by AdaIN)
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True)
        )

        # Load pretrained VGG weights
        self.vgg.load_state_dict(
            torch.load(vgg_path, map_location=device)
        )

        layers = list(self.vgg.children())

        self.enc_1 = nn.Sequential(*layers[:4])
        self.enc_2 = nn.Sequential(*layers[4:11])
        self.enc_3 = nn.Sequential(*layers[11:18])
        self.enc_4 = nn.Sequential(*layers[18:31])

        # Freeze encoder
        for block in [self.enc_1, self.enc_2, self.enc_3, self.enc_4]:
            for param in block.parameters():
                param.requires_grad = False


    def forward(self, x):
        x = self.enc_1(x)
        x = self.enc_2(x)
        x = self.enc_3(x)
        x = self.enc_4(x)

        return x
