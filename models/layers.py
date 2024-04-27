
import torch
import torch.nn as nn

from torch.nn import functional as F
import numpy as np  

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,padding, dilation, groups, bias)

        if kernel_size == 1:
            self.ind = True   # kernel=1 的话 直接就F.conv
        else:
            self.ind = False            
            self.out_channels = out_channels  # 256
            self.ks = kernel_size  # 3
            
            self.avg_pool = nn.AdaptiveAvgPool2d((kernel_size, kernel_size))                  # 3x3 的 avgPool
            self.ce_fg = nn.Linear(kernel_size*kernel_size, kernel_size*kernel_size, False)   # Linear: 9 => 9
            self.ce_bg = nn.Linear(kernel_size*kernel_size, kernel_size*kernel_size, False)   # Linear: 9 => 9

            self.act = nn.ReLU(inplace=True)
            self.sig = nn.Sigmoid()
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)         # nn.Unfold(3,1,0,1)
    
    def forward(self, x, bg_proto, fg_proto):    # [b_q,256,16,16] / [1,256] / [1,256]
        if self.ind:
            return F.conv2d(x, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        else:
            b, c, _, _ = x.size()  # [b_q,256,h,w]
            weight = self.weight   # [256,256,3,3]

            gl = self.avg_pool(x)  # [b_q,256,3,3]
            gl_fg_sim = F.cosine_similarity(gl, fg_proto[..., None, None], dim=1).unsqueeze(dim=1)  # [b_q,1,3,3]
            gl_fg = (gl * gl_fg_sim).view(b,c,-1)   # [b_q,256,3*3]
            gl_fg = self.act(self.ce_fg(gl_fg)).view(b,1,c,self.ks,self.ks)  # [b_q,1,256,3,3]

            gl_bg_sim = (1.0 - F.cosine_similarity(gl, bg_proto[..., None, None], dim=1)).unsqueeze(dim=1)  # [b_q,1,3,3]
            gl_bg = (gl * gl_bg_sim).view(b,c,-1)   # [b_q,256,3*3]
            gl_bg = self.act(self.ce_bg(gl_bg)).view(b,c,1,self.ks,self.ks)  # [b_q,256,1,3,3]
            out = self.sig(gl_fg + gl_bg)  # [b_q,256,256,3,3]

            x_un = self.unfold(x)  # [b_q,256*3*3,h*w]
            l = x_un.shape[-1]     # h*w
            out = (out * weight.unsqueeze(0)).view(b, self.out_channels, -1)  # [b_q,256,256*3*3]

            return torch.matmul(out, x_un).view(b, self.out_channels, int(np.sqrt(l)), int(np.sqrt(l)))   # [b_q,256,16,16]


