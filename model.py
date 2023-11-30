import torch
import torch.nn.functional as F
from torch import nn

from networks.unet_model import UNet
from networks.cross_attn import CrossEncoder, Decoder
from networks.extractor import BasicEncoder
from networks.seg import U2NETP


class CAM_Module(nn.Module):
    # Reference: https://github.com/yearing1017/DANet_PyTorch/blob/master/DAN_ResNet/attention.py
    """ Channel attention module"""

    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input mask maps( B X C X H X W)
            returns :
                out : attention value
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)

        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, scale=8):
        super(UpdateBlock, self).__init__()
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, scale * scale * 9, 1, padding=0))

    def forward(self, imgf, coords1):
        mask = 0.25 * self.mask(imgf)  # scale mask to balence gradients
        dflow = self.flow_head(imgf)
        coords1 = coords1 + dflow

        return mask, coords1


def coords_grid(batch, ht, wd, gap=1):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[:, ::gap, ::gap]
    return coords[None].repeat(batch, 1, 1, 1)


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        # print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        # print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


class Seg(nn.Module):
    def __init__(self):
        super(Seg, self).__init__()
        self.msk = U2NETP(3, 1)

    def forward(self, x):
        d0, hx6, hx5d, hx4d, hx3d, hx2d, hx1d = self.msk(x)
        return d0, hx6, hx5d, hx4d, hx3d, hx2d, hx1d


class DewarpTextlineMaskCrossAtten(nn.Module):
    def __init__(self, image_size=224, hdim=256):
        super(DewarpTextlineMaskCrossAtten, self).__init__()
        self.image_size = image_size
        self.hdim = hdim
        self.n_head = 8
        self.d_v = self.hdim // self.n_head
        self.d_k = self.hdim // self.n_head
        self.basic_net = BasicEncoder(in_channels=3, output_dim=self.hdim)

        self.encoder = CrossEncoder(n_layers=12, n_head=self.n_head, d_model=self.hdim, d_k=self.d_k, d_v=self.d_v,
                                    d_inner=2048, n_position=self.image_size // 8)
        self.decoder = Decoder(n_layers=6, n_head=self.n_head, d_model=self.hdim * 2, d_k=self.d_k * 2,
                               d_v=self.d_v * 2,
                               d_inner=2048, n_position=self.image_size // 8)

        self.cam_1 = CAM_Module()
        self.cam_2 = CAM_Module()
        self.cam_3 = CAM_Module()

        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels=64 * 6, out_channels=self.hdim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hdim)
        )

        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=self.hdim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.hdim)
        )

        self.update_block = UpdateBlock(self.hdim * 2)

    def _upsample(self, x, size):
        _, _, H, W = size
        return F.upsample(x, size=(H, W), mode='bilinear')  # , align_corners=False)

    def initialize_flow(self, img):
        N, C, H, W = img.shape
        coodslar = coords_grid(N, H, W).to(img.device)
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        return coodslar, coords0, coords1

    def upsample_flow(self, flow, mask):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image, fore_mask, textline_mask):
        fmap = self.basic_net(image)
        fmap = self.cam_1(fmap)

        fore_mask = self.conv3x3_1(fore_mask)
        fore_mask = self.cam_2(fore_mask)

        textline_mask = self.conv3x3_2(textline_mask)
        textline_mask = F.interpolate(textline_mask, size=fmap.shape[2:], mode='bilinear', align_corners=False)
        textline_mask = self.cam_3(textline_mask)

        n, c, h, w = fore_mask.size()
        output_mask = self.encoder(fmap, fore_mask)
        output_mask = output_mask.transpose(1, 2).contiguous().view(n, c, h, w)

        n, c, h, w = textline_mask.size()
        output_textline_mask = self.encoder(fmap, textline_mask)
        output_textline_mask = output_textline_mask.transpose(1, 2).contiguous().view(n, c, h, w)

        output = torch.cat((output_mask, output_textline_mask), dim=1)

        output = self.decoder(output)
        outmap = output.transpose(1, 2).contiguous().view(n, c * 2, h, w)

        coodslar, coords0, coords1 = self.initialize_flow(image)
        coords1 = coords1.detach()

        mask, coords1 = self.update_block(outmap, coords1)
        flow_up = self.upsample_flow(coords1 - coords0, mask)
        bm_up = coodslar + flow_up

        return bm_up


class DewarpTextlineMaskGuide(nn.Module):
    def __init__(self, image_size=256):
        super(DewarpTextlineMaskGuide, self).__init__()
        self.hdim = 256
        self.dewarp_net = DewarpTextlineMaskCrossAtten(image_size=image_size, hdim=self.hdim)

        self.initialize_weights_()
        self.seg = Seg()
        self.unet = UNet(n_channels=3, n_classes=1)

    def initialize_weights_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                torch.nn.init.xavier_normal_(m.weight, gain=0.2)
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, image):
        d0, hx6, hx5d, hx4d, hx3d, hx2d, hx1d = self.seg(image)
        hx6 = F.interpolate(hx6, scale_factor=4, mode='bilinear', align_corners=False)
        hx5d = F.interpolate(hx5d, scale_factor=2, mode='bilinear', align_corners=False)
        hx4d = F.interpolate(hx4d, scale_factor=1, mode='bilinear', align_corners=False)
        hx3d = F.interpolate(hx3d, scale_factor=0.5, mode='bilinear', align_corners=False)
        hx2d = F.interpolate(hx2d, scale_factor=0.25, mode='bilinear', align_corners=False)
        hx1d = F.interpolate(hx1d, scale_factor=0.125, mode='bilinear', align_corners=False)

        seg_map_all = torch.cat((hx6, hx5d, hx4d, hx3d, hx2d, hx1d), dim=1)
        textline_map, textline_mask = self.unet(image)

        bm_up = self.dewarp_net(image, seg_map_all, textline_map)

        return bm_up


if __name__ == '__main__':
    x = torch.randn((2, 3, 224, 224)).cuda()

    net = DewarpTextlineMaskGuide(image_size=x.shape[-1]).cuda()

    checkpoint_path = 'pretrained_models/30.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(checkpoint)

    print(net)
    pred = net(x)
    n_p = sum(x.numel() for x in net.parameters())  # number parameters
    n_g = sum(x.numel() for x in net.parameters() if x.requires_grad)  # number gradients
    print('Model Summary: %g parameters, %g gradients\n' % (n_p, n_g))

    print("pred: ", pred.shape)

