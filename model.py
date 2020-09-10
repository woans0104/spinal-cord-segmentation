import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#########################################################################################################


# 2d model

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, momentum=0.1, coordconv=False,
                 radius=False):
        super(ConvBnRelu, self).__init__()


        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  padding=padding, stride=stride)

        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackEncoder, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3),
                                 stride=1, padding=padding, momentum=momentum)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace


class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, momentum=0.1, coordconv=False, radius=False):
        super(StackDecoder, self).__init__()

        # self.upSample = nn.Upsample(size=upsample_size, scale_factor=(2,2), mode='bilinear')
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum, coordconv=coordconv, radius=radius)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 momentum=momentum)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):

        x = self.transpose_conv(x)
        if down_tensor != None:
            x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)
        return x


class Conv2d_ws(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_ws, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvTranspose2d_ws(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super(ConvTranspose2d_ws, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, output_padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight

        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)

        return F.conv_transpose2d(x, weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)



###########################################################################################################################

class Conv_nomarlize_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride,group_channel=1,affine=True,nomalize_con='gn',weight_std=False):
        super(Conv_nomarlize_Relu, self).__init__()
        self.nomalize_con = nomalize_con
        self.conv = Conv2d_ws if weight_std else nn.Conv2d


        self.conv1 = self.conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

        if self.nomalize_con == 'gn':
            self.nomalize = nn.GroupNorm(group_channel,out_channels,affine=affine)

        elif self.nomalize_con == 'in':
            self.nomalize = nn.InstanceNorm2d(out_channels,affine=affine)
        else:
            print('bnbnbn')
            self.nomalize = nn.BatchNorm2d(out_channels,affine=affine)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.nomalize(x)
        x = self.relu(x)
        return x

class StackEncoder_norm(nn.Module):
    def __init__(self, in_channels, out_channels, padding, group_channel=1,affine=True,nomalize_con='gn',weight_std=False):
        super(StackEncoder_norm, self).__init__()
        self.convr1 = Conv_nomarlize_Relu(in_channels, out_channels, kernel_size=(3, 3),stride=1, padding=padding,
                                 group_channel= group_channel,affine=affine,nomalize_con=nomalize_con,weight_std=weight_std)

        self.convr2 = Conv_nomarlize_Relu(out_channels, out_channels, kernel_size=(3, 3),stride=1, padding=padding,
                                 group_channel= group_channel,affine=affine,nomalize_con=nomalize_con,weight_std=weight_std)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace


class StackDecoder_norm(nn.Module):
    def __init__(self, in_channels, out_channels, padding, group_channel=1,affine=True,nomalize_con='gn',weight_std=False):
        super(StackDecoder_norm, self).__init__()


        if weight_std:
            self.transpose_conv = ConvTranspose2d_ws(in_channels, out_channels, kernel_size=(2, 2), stride=2)
        else:
            self.transpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2)

        self.convr1 = Conv_nomarlize_Relu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 group_channel=group_channel,affine=affine,nomalize_con=nomalize_con,weight_std=weight_std)
        self.convr2 = Conv_nomarlize_Relu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding,
                                 group_channel= group_channel,affine=affine,nomalize_con=nomalize_con,weight_std=weight_std)

    def _crop_concat(self, upsampled, bypass):

        margin = bypass.size()[2] - upsampled.size()[2]
        c = margin // 2
        if margin % 2 == 1:
            bypass = F.pad(bypass, (-c, -c - 1, -c, -c - 1))
        else:
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):

        x = self.transpose_conv(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)
        return x




######################################################################################################
# model




class Unet2D_spinal(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding=1, momentum=0.1,start_channel=64):
        super(Unet2D_spinal, self).__init__()
        self.channels, self.heights, self.width = in_shape
        self.padding = padding
        self.start_channel = start_channel


        self.down1 = StackEncoder(self.channels, self.start_channel, padding, momentum=momentum)
        self.down2 = StackEncoder(self.start_channel, self.start_channel*2, padding, momentum=momentum)
        self.down3 = StackEncoder(self.start_channel*2, self.start_channel*4, padding, momentum=momentum)


        self.center = nn.Sequential(
            ConvBnRelu(self.start_channel*4, self.start_channel*8, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(self.start_channel*8, self.start_channel*8, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )

        self.up1 = StackDecoder(in_channels=self.start_channel*8, out_channels=self.start_channel*4, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=self.start_channel*4, out_channels=self.start_channel*2, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=self.start_channel*2, out_channels=self.start_channel, padding=padding, momentum=momentum)


        self.output_seg_map = nn.Conv2d(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(self.heights, self.width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)


        center = self.center(x)

        x = self.up1(center, x_trace3)
        x = self.up2(x, x_trace2)
        x = self.up3(x, x_trace1)

        out = self.output_seg_map(x)

        if out.shape[-1] != self.width:
            out = self.output_up_seg_map(out)

        return out,center


class Unet2D_spinal_norm(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding=1, nomalize_con='gn', affine=True,group_channel=1,weight_std=False,start_channel=64):

        super(Unet2D_spinal_norm, self).__init__()
        self.channels, self.heights, self.width = in_shape
        self.start_channel = start_channel

        self.down1 = StackEncoder_norm(self.channels, self.start_channel, padding, group_channel=group_channel,
                                       affine=affine, nomalize_con=nomalize_con, weight_std=weight_std)
        self.down2 = StackEncoder_norm(self.start_channel, self.start_channel*2, padding,group_channel=group_channel,
                                       affine=affine,nomalize_con=nomalize_con,weight_std=weight_std)
        self.down3 = StackEncoder_norm(self.start_channel*2, self.start_channel*4, padding,group_channel=group_channel,
                                       affine=affine,nomalize_con=nomalize_con,weight_std=weight_std)


        self.center = nn.Sequential(
            Conv_nomarlize_Relu(self.start_channel*4, self.start_channel*8, kernel_size=(3, 3), stride=1, padding=padding,
                                group_channel=group_channel,affine=affine,nomalize_con=nomalize_con,weight_std=weight_std),
            Conv_nomarlize_Relu(self.start_channel*8, self.start_channel*8, kernel_size=(3, 3), stride=1, padding=padding,
                                group_channel=group_channel,affine=affine,nomalize_con=nomalize_con,weight_std=weight_std),
        )

        self.up1 = StackDecoder_norm(in_channels=self.start_channel*8, out_channels=self.start_channel*4, padding=padding,
                                     group_channel=group_channel,affine=affine,nomalize_con=nomalize_con,weight_std=weight_std)
        self.up2 = StackDecoder_norm(in_channels=self.start_channel*4, out_channels=self.start_channel*2, padding=padding,
                                     group_channel=group_channel,affine=affine,nomalize_con=nomalize_con,weight_std=weight_std)
        self.up3 = StackDecoder_norm(in_channels=self.start_channel*2, out_channels=self.start_channel, padding=padding,
                                     group_channel=group_channel,affine=affine,nomalize_con=nomalize_con,weight_std=weight_std)

        if weight_std:
            self.output_seg_map = Conv2d_ws(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)
        else:
            self.output_seg_map = nn.Conv2d(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)


        self.output_up_seg_map = nn.Upsample(size=(self.heights, self.width), mode='nearest')

    def forward(self, x):
        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)


        center = self.center(x)

        x = self.up1(center, x_trace3)
        x = self.up2(x, x_trace2)
        x = self.up3(x, x_trace1)

        out = self.output_seg_map(x)

        if out.shape[-1] != self.width:
            out = self.output_up_seg_map(out)

        return out,center





class ae_spinal(nn.Module):
    def __init__(self,in_shape,start_channel=64,kenel_size=3,padding=1):
        super(ae_spinal, self).__init__()

        self.channels, self.heights, self.width = in_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(1, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel * 2, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            #center
            nn.Conv2d(start_channel * 4, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 8),
            nn.ReLU(),

        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(start_channel*8, start_channel*8, 2, stride=2),
            nn.Conv2d(start_channel * 8, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*4),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel * 4, start_channel*4, 2, stride=2),
            nn.Conv2d(start_channel * 4, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel*2 , start_channel*2, 2, stride=2),
            nn.Conv2d(start_channel * 2, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),

            nn.Conv2d(start_channel, 1, 1)

        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)

        return decoder,encoder




##########################################################################

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a*b , -1)
        G = torch.mm(features, features.t())  # compute the gram product

        return G.div(a * b * c * d)



class Unet2D_style_spinal(nn.Module):
    """
    ori unet : padding =1, momentum=0.1,coordconv =False
    down_crop unet : padding =0, momentum=0.1,coordconv =False
    """

    def __init__(self, in_shape, padding=1, momentum=0.1,start_channel=64,style=3):
        super(Unet2D_style_spinal, self).__init__()
        self.channels, self.heights, self.width = in_shape
        self.padding = padding
        self.start_channel = start_channel
        self.style = style



        self.down1 = StackEncoder(self.channels, self.start_channel, padding, momentum=momentum)
        self.down2 = StackEncoder(self.start_channel, self.start_channel*2, padding, momentum=momentum)
        self.down3 = StackEncoder(self.start_channel*2, self.start_channel*4, padding, momentum=momentum)


        self.center = nn.Sequential(
            ConvBnRelu(self.start_channel*4, self.start_channel*8, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum),
            ConvBnRelu(self.start_channel*8, self.start_channel*8, kernel_size=(3, 3), stride=1, padding=padding, momentum=momentum)
        )


        self.up1 = StackDecoder(in_channels=self.start_channel*8, out_channels=self.start_channel*4, padding=padding, momentum=momentum)
        self.up2 = StackDecoder(in_channels=self.start_channel*4, out_channels=self.start_channel*2, padding=padding, momentum=momentum)
        self.up3 = StackDecoder(in_channels=self.start_channel*2, out_channels=self.start_channel, padding=padding, momentum=momentum)

        self.output_seg_map = nn.Conv2d(self.start_channel, 1, kernel_size=(1, 1), padding=0, stride=1)
        self.output_up_seg_map = nn.Upsample(size=(self.heights, self.width), mode='nearest')



    def forward(self, x):


        x, x_trace1 = self.down1(x)
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)

        center = self.center(x)


        x = self.up1(center, x_trace3)
        x = self.up2(x, x_trace2)
        x = self.up3(x, x_trace1)

        out = self.output_seg_map(x)


        if self.style ==3 :
            return out, [x_trace3,center]


        elif self.style == 4:
            return out, [center]




class ae_spinal_style(nn.Module):
    # down 4
    def __init__(self,in_shape,start_channel=64,kenel_size=3,padding=1,style=1):
        super(ae_spinal_style, self).__init__()

        self.channels, self.heights, self.width = in_shape
        self.style = style



        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),

        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 2),
            nn.ReLU(),

        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(start_channel * 2, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 4),
            nn.ReLU(),


        )


        self.center = nn.Sequential(

            nn.MaxPool2d(2, stride=2),
            #center
            nn.Conv2d(start_channel * 4, start_channel * 8, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 8),
            nn.ReLU(),

        )

        self.decoder = nn.Sequential(


            nn.ConvTranspose2d(start_channel * 8, start_channel * 8, 2, stride=2),
            nn.Conv2d(start_channel * 8, start_channel * 4, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel * 4, start_channel*4, 2, stride=2),
            nn.Conv2d(start_channel * 4, start_channel * 2, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel*2),
            nn.ReLU(),

            nn.ConvTranspose2d(start_channel*2 , start_channel*2, 2, stride=2),
            nn.Conv2d(start_channel * 2, start_channel, kenel_size, padding=padding),
            nn.BatchNorm2d(start_channel),
            nn.ReLU(),



            nn.Conv2d(start_channel, 1, 1)

        )



    def forward(self, x):


        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)
        center = self.center(encoder3)

        decoder = self.decoder(center)


        if self.style == 3:

            return decoder, [encoder3,center]

        elif self.style == 4:

            return decoder, [center]







##########################################################################

if __name__ == '__main__':
    from torchsummary import summary

    my_net = Unet2D_style_spinal(in_shape=(1, 128, 128))

    summary(model=my_net.cuda(),input_size=(1,128,128))



