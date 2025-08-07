from attention.SENet import *

class CACB_Module(nn.Module):
    def __init__(self,in_ch,out_ch,last :True):
        super(CACB_Module,self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        cbr = nn.Sequential(nn.Conv2d(in_channels=self.in_ch,out_channels=self.out_ch,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(self.out_ch),
                                 )

        if last:
            self.cbr = nn.Sequential(cbr,nn.Tanh())

        else:
            self.cbr = nn.Sequential(cbr,nn.LeakyReLU(0.2))

        self.SE = SELayer(channel=self.in_ch)

    def forward(self,x):
        x_attention = self.SE(x)
        out = self.cbr(x_attention)

        return out

if __name__ == '__main__':
    x = torch.randn(1,32,256,256)
    layer = CACB_Module(in_ch=32,out_ch=3,last=False)
    y = layer(x)
    print(y.shape)