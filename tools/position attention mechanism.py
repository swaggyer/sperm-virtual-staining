import torch
import torch.nn as nn


class PAM(nn.Module):
    def __init__(self,in_ch):
        super(PAM,self).__init__()
        self.in_ch = in_ch


    def forward(self,x):
        h = x.shape[-2]
        w = x.shape[-1]
        c = x.shape[-3]
        b = x.shape[0]

        xb = x.view(b,c,h*w)
        xb = xb.permute(0,2,1)
        xc = x.view(b,c,h*w)
        xd = xc
        xs = torch.softmax(torch.bmm(xb,xc),dim=-1)
        xe = torch.bmm(xd,xs)



        return xe

if __name__ == '__main__':
    x = torch.randn(2,3,4,4)
    layer = PAM(3)
    y = layer(x)
    print(y.shape)
    # print(y)

