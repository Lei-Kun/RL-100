import torch
import torch.nn.functional as F
import torch.nn as nn

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x, generator=None):
        # import pdb; pdb.set_trace()
        if x.size(3) != 3:
            x = x.permute(0, 2, 3, 1)  # to [n, h, w, c]
        n, h, w, c = x.size()
        
        assert c == 3
        padding = tuple([self.pad] * 4)
        
        # tansfer to [n, c, h, w]
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, padding, 'replicate')

        # 分别为高度和宽度创建坐标网格
        eps_h = 1.0 / (h + 2 * self.pad)
        eps_w = 1.0 / (w + 2 * self.pad)
        
        # 创建高度方向的坐标
        arange_h = torch.linspace(-1.0 + eps_h,
                                  1.0 - eps_h,
                                  h + 2 * self.pad,
                                  device=x.device,
                                  dtype=x.dtype)[:h]
        
        # 创建宽度方向的坐标
        arange_w = torch.linspace(-1.0 + eps_w,
                                  1.0 - eps_w,
                                  w + 2 * self.pad,
                                  device=x.device,
                                  dtype=x.dtype)[:w]
        
        # 构建网格坐标 (h, w, 2)，其中最后一维是 [x, y] 坐标
        grid_x = arange_w.unsqueeze(0).repeat(h, 1)  # (h, w)
        grid_y = arange_h.unsqueeze(1).repeat(1, w)  # (h, w)
        base_grid = torch.stack([grid_x, grid_y], dim=2)  # (h, w, 2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)  # (n, h, w, 2)

        # 生成随机偏移，分别对应宽度和高度方向
        shift_x = torch.randint(0,
                               2 * self.pad + 1,
                               size=(n, 1, 1, 1),
                               device=x.device,
                               dtype=x.dtype,
                               generator=generator) * 2.0 / (w + 2 * self.pad)
        
        shift_y = torch.randint(0,
                               2 * self.pad + 1,
                               size=(n, 1, 1, 1),
                               device=x.device,
                               dtype=x.dtype,
                               generator=generator) * 2.0 / (h + 2 * self.pad)
        
        shift = torch.cat([shift_x, shift_y], dim=3)  # (n, 1, 1, 2)

        grid = base_grid + shift

        x = F.grid_sample(x,
                          grid,
                          padding_mode='zeros',
                          align_corners=False)
        
        # transfer back to [n, h, w, c]
        x = x.permute(0, 2, 3, 1)
        
        return x
