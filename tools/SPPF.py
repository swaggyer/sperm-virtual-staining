from attention.SENet import *

class SPPF(nn.Module):
    def __init__(self,in_chs,out_chs):
        super(SPPF,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out


class SPPF_avg555(nn.Module):
    def __init__(self,in_chs,out_chs):
        super(SPPF_avg555,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.AvgPool2d(kernel_size=5,stride=1,padding=2)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out

class SPPF_avg333(nn.Module):  ##池化核的大小 3 3 3
    def __init__(self,in_chs,out_chs):
        super(SPPF_avg333,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))


    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)

        out = self.CBA2(features)

        return out

class SPPF333(nn.Module):  ##池化核的大小 3 3 3
    def __init__(self,in_chs,out_chs):
        super(SPPF333,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))


    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)

        out = self.CBA2(features)

        return out


class SPPF777(nn.Module):  ##池化核的大小 7 7 7
    def __init__(self,in_chs,out_chs):
        super(SPPF777,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.MaxPool2d(kernel_size=7,stride=1,padding=3)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out

class SPPF_avg777(nn.Module):  ##池化核的大小 7 7 7
    def __init__(self,in_chs,out_chs):
        super(SPPF_avg777,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.AvgPool2d(kernel_size=7,stride=1,padding=3)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out

class SPPF777_atten(nn.Module):  ##池化核的大小 7 7 7
    def __init__(self,in_chs,out_chs):
        super(SPPF777_atten,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.MaxPool2d(kernel_size=7,stride=1,padding=3)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

        self.atten = SELayer(self.in_chs*2)

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        features = self.atten(features)
        out = self.CBA2(features)

        return out


if __name__ == '__main__':
    x = torch.randn(1,512,32,32)
    layer = SPPF777_atten(512,256)
    y = layer(x)
    print(y.shape)


class SPPF999(nn.Module):  ##池化核的大小 7 7 7
    def __init__(self,in_chs,out_chs):
        super(SPPF999,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.MaxPool2d(kernel_size=9,stride=1,padding=4)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out

class SPPF_avg999(nn.Module):  ##池化核的大小 7 7 7
    def __init__(self,in_chs,out_chs):
        super(SPPF_avg999,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.AvgPool2d(kernel_size=9,stride=1,padding=4)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out

class SPPF11(nn.Module):  ##池化核的大小 7 7 7
    def __init__(self,in_chs,out_chs):
        super(SPPF11,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.MaxPool2d(kernel_size=11,stride=1,padding=5)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out


class SPPF_avg11(nn.Module):  ##池化核的大小 7 7 7
    def __init__(self,in_chs,out_chs):
        super(SPPF_avg11,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool = nn.AvgPool2d(kernel_size=11,stride=1,padding=5)
        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool(f1)
        f3 = self.pool(f2)
        f4 = self.pool(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out

class SPPF357(nn.Module):  ##池化核的大小 3 3 3
    def __init__(self,in_chs,out_chs):
        super(SPPF357,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=7,stride=1,padding=3)

        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool1(f1)
        f3 = self.pool2(f2)
        f4 = self.pool3(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out


class SPPF_avg357(nn.Module):
    def __init__(self,in_chs,out_chs):
        super(SPPF_avg357,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool1 = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=5,stride=1,padding=2)
        self.pool3 = nn.AvgPool2d(kernel_size=7,stride=1,padding=3)

        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool1(f1)
        f3 = self.pool2(f2)
        f4 = self.pool3(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out



class SPPF579(nn.Module):  ##池化核的大小 3 3 3
    def __init__(self,in_chs,out_chs):
        super(SPPF579,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool1 = nn.MaxPool2d(kernel_size=5,stride=1,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=7,stride=1,padding=3)
        self.pool3 = nn.MaxPool2d(kernel_size=9,stride=1,padding=4)

        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool1(f1)
        f3 = self.pool2(f2)
        f4 = self.pool3(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out

class SPPF_avg579(nn.Module):  ##池化核的大小 3 3 3
    def __init__(self,in_chs,out_chs):
        super(SPPF_avg579,self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.pool1 = nn.AvgPool2d(kernel_size=5,stride=1,padding=2)
        self.pool2 = nn.AvgPool2d(kernel_size=7,stride=1,padding=3)
        self.pool3 = nn.AvgPool2d(kernel_size=9,stride=1,padding=4)

        self.CBA1 = nn.Sequential(nn.Conv2d(in_channels=self.in_chs,out_channels=(self.in_chs//2),kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.in_chs//2),
                                 nn.LeakyReLU(0.2))
        self.CBA2= nn.Sequential(nn.Conv2d(in_channels=(self.in_chs*2),out_channels=self.out_chs,kernel_size=1,stride=1,padding=0),
                                 nn.BatchNorm2d(self.out_chs),
                                 nn.LeakyReLU(0.2))

    def forward(self,x):
        f1 = self.CBA1(x)
        f2 = self.pool1(f1)
        f3 = self.pool2(f2)
        f4 = self.pool3(f3)
        features = torch.cat((f1,f2,f3,f4),dim = 1)
        out = self.CBA2(features)

        return out