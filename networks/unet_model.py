from networks.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return x, logits


if __name__ == '__main__':
    x1 = torch.rand((2, 3, 224, 224)).cuda()
    net = UNet(n_channels=3, n_classes=1).cuda()
    print(net)
    map, pred_img = net(x1)
    n_p = sum(x.numel() for x in net.parameters())  # number parameters
    n_g = sum(x.numel() for x in net.parameters() if x.requires_grad)  # number gradients
    print('Model Summary: %g parameters, %g gradients\n' % (n_p, n_g))

    print("map: ", map.shape)
    print("pred_img: ", pred_img.shape)