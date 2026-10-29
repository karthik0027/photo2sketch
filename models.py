import torch
import torch.nn as nn


def down(in_c, out_c, norm=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up(in_c, out_c, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_c),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, nf=64):
        super().__init__()
        chs = [nf, nf * 2, nf * 4, nf * 8, nf * 8, nf * 8, nf * 8, nf * 8]

        self.downs = nn.ModuleList()
        prev = in_ch
        for i, c in enumerate(chs):
            self.downs.append(down(prev, c, norm=(0 < i < 7)))
            prev = c

        self.ups = nn.ModuleList(
            [
                up(chs[7], chs[6], dropout=True),
                up(chs[6] * 2, chs[5], dropout=True),
                up(chs[5] * 2, chs[4], dropout=True),
                up(chs[4] * 2, chs[3]),
                up(chs[3] * 2, chs[2]),
                up(chs[2] * 2, chs[1]),
                up(chs[1] * 2, chs[0]),
            ]
        )
        self.final = nn.Sequential(nn.ConvTranspose2d(chs[0] * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
        x = skips[-1]
        for i, u in enumerate(self.ups):
            x = u(x)
            x = torch.cat([x, skips[-2 - i]], dim=1)
        return self.final(x)


class PatchDiscriminator(nn.Module):

    def __init__(self, in_ch=4, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, 1, 4, 1, 1),
        )

    def forward(self, photo, sketch):
        return self.net(torch.cat([photo, sketch], dim=1))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
