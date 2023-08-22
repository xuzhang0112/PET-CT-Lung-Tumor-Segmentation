from torch import nn
import torch
import os


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encode(x)


class FusionDecode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.concat_conv = DoubleConv(in_channels, in_channels//2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, up_sample_feature, concat_feature):
        up_sample_feature = self.up_sample(up_sample_feature)
        concat_feature = self.concat_conv(concat_feature)
        x = self.double_conv(torch.concat(
            [concat_feature, up_sample_feature], dim=1))
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class YNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes  # 输出类别数
        # pet encoding
        self.pet_encode1 = DoubleConv(1, 16)
        self.pet_encode2 = Encode(16, 32)
        self.pet_encode3 = Encode(32, 64)
        self.pet_encode4 = Encode(64, 128)
        self.pet_encode5 = Encode(128, 256)
        # ct encoding
        self.ct_encode1 = DoubleConv(1, 16)
        self.ct_encode2 = Encode(16, 32)
        self.ct_encode3 = Encode(32, 64)
        self.ct_encode4 = Encode(64, 128)
        self.ct_encode5 = Encode(128, 256)
        # fusion branch
        self.fuse5 = DoubleConv(512, 256)
        self.fuse4 = FusionDecode(256, 128)
        self.fuse3 = FusionDecode(128, 64)
        self.fuse2 = FusionDecode(64, 32)
        self.fuse1 = FusionDecode(32, 16)
        # pet output
        self.pet_output = OutConv(16, 2)
        # ct output
        self.ct_output = OutConv(16, 2)

    def forward(self, pet_image, ct_image):
        # pet encoding
        pet_feature1 = self.pet_encode1(pet_image)
        pet_feature2 = self.pet_encode2(pet_feature1)
        pet_feature3 = self.pet_encode3(pet_feature2)
        pet_feature4 = self.pet_encode4(pet_feature3)
        pet_feature5 = self.pet_encode5(pet_feature4)
        # ct encoding
        ct_feature1 = self.ct_encode1(ct_image)
        ct_feature2 = self.ct_encode2(ct_feature1)
        ct_feature3 = self.ct_encode3(ct_feature2)
        ct_feature4 = self.ct_encode4(ct_feature3)
        ct_feature5 = self.ct_encode5(ct_feature4)
        # fusion branch
        fuse_feature5 = self.fuse5(
            torch.concat((pet_feature5, ct_feature5), dim=1))
        fuse_feature4 = self.fuse4(
            fuse_feature5, torch.concat((pet_feature4, ct_feature4), dim=1))
        fuse_feature3 = self.fuse3(
            fuse_feature4, torch.concat((pet_feature3, ct_feature3), dim=1))
        fuse_feature2 = self.fuse2(
            fuse_feature3, torch.concat((pet_feature2, ct_feature2), dim=1))
        fuse_feature1 = self.fuse1(
            fuse_feature2, torch.concat((pet_feature1, ct_feature1), dim=1))
        # pet segmentation output
        pet_output = self.pet_output(fuse_feature1)
        # ct segmentation output
        ct_output = self.ct_output(fuse_feature1)
        return pet_output, ct_output


class CrossModalityPositionAttention(nn.Module):
    def __init__(self, channels, scale_coefficient=4):
        super().__init__()
        self.q_conv = Conv(channels, channels//scale_coefficient)
        self.k_conv = Conv(channels, channels//scale_coefficient)
        self.v_conv = Conv(channels, channels//scale_coefficient)
        self.activation = nn.Softmax(dim=-1)
        self.res_conv = Conv(channels//scale_coefficient, channels)

    def forward(self, feature1, feature2):
        q = self.q_conv(feature2)
        k = self.k_conv(feature1)
        v = self.v_conv(feature1)
        b, c, w, h = q.shape
        q = q.reshape(b, c, w*h)
        q = q.transpose(1, 2)
        k = k.reshape(b, c, w*h)
        v = v.reshape(b, c, w*h)
        v = v.transpose(1, 2)
        # print(q.shape, k.shape, v.shape)
        # print(attention_map.shape, v.shape)
        f = torch.bmm(self.activation(torch.bmm(q, k)), v)
        f = f.transpose(1, 2)
        f = f.reshape(b, c, w, h)
        f = self.res_conv(f)
        return feature1+f


class CrossModalityChannelAttention(nn.Module):
    def __init__(self, channels, scale_coefficient=4):
        super().__init__()
        self.q_conv = Conv(channels, channels//scale_coefficient)
        self.k_conv = Conv(channels, channels//scale_coefficient)
        self.v_conv = Conv(channels, channels//scale_coefficient)
        self.activation = nn.Softmax(dim=-1)
        self.res_conv = Conv(channels//scale_coefficient, channels)

    def forward(self, feature1, feature2):
        q = self.q_conv(feature2)
        k = self.k_conv(feature1)
        v = self.v_conv(feature1)
        b, c, w, h = q.shape
        q = q.reshape(b, c, w*h)
        k = k.reshape(b, c, w*h)
        k = k.transpose(1, 2)
        v = v.reshape(b, c, w*h)
        f = torch.bmm(self.activation(torch.bmm(q, k)), v)
        f = f.reshape(b, c, w, h)
        f = self.res_conv(f)
        return feature1+f


class CrossModalityDecode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.cross_modality_position_attention = CrossModalityPositionAttention(
            in_channels//2)
        self.cross_modality_channel_attention = CrossModalityChannelAttention(
            in_channels//2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, feature_, feature1, feature2):
        up_sample_feature = self.up_sample(feature_)
        concat_feature = self.cross_modality_position_attention(
            feature1, feature2)+self.cross_modality_channel_attention(feature1, feature2)
        x = self.double_conv(torch.concat(
            [concat_feature, up_sample_feature], dim=1))
        return x


class ModalitySpecificDecode(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, feature_, concat_feature):
        up_sample_feature = self.up_sample(feature_)
        x = self.double_conv(torch.concat(
            [concat_feature, up_sample_feature], dim=1))
        return x


class XNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes  # 输出类别数
        # pet encoding
        self.pet_encode1 = DoubleConv(1, 16)
        self.pet_encode2 = Encode(16, 32)
        self.pet_encode3 = Encode(32, 64)
        self.pet_encode4 = Encode(64, 128)
        self.pet_encode5 = Encode(128, 256)
        # ct encoding
        self.ct_encode1 = DoubleConv(1, 16)
        self.ct_encode2 = Encode(16, 32)
        self.ct_encode3 = Encode(32, 64)
        self.ct_encode4 = Encode(64, 128)
        self.ct_encode5 = Encode(128, 256)
        # fusion branch
        self.fuse5 = DoubleConv(512, 256)
        self.fuse4 = FusionDecode(256, 128)
        self.fuse3 = FusionDecode(128, 64)
        self.fuse2 = FusionDecode(64, 32)
        self.fuse1 = FusionDecode(32, 16)
        # pet modality-specific decoding
        self.pet_decode4 = ModalitySpecificDecode(256, 128)
        self.pet_decode3 = ModalitySpecificDecode(128, 64)
        self.pet_decode2 = ModalitySpecificDecode(64, 32)
        self.pet_decode1 = ModalitySpecificDecode(32, 16)
        # ct modality-specific decoding
        self.ct_decode4 = ModalitySpecificDecode(256, 128)
        self.ct_decode3 = ModalitySpecificDecode(128, 64)
        self.ct_decode2 = ModalitySpecificDecode(64, 32)
        self.ct_decode1 = ModalitySpecificDecode(32, 16)
        # pet output
        self.pet_output = OutConv(32, 2)
        # ct output
        self.ct_output = OutConv(32, 2)

    def forward(self, pet_image, ct_image):
        # pet encoding
        pet_feature1 = self.pet_encode1(pet_image)
        pet_feature2 = self.pet_encode2(pet_feature1)
        pet_feature3 = self.pet_encode3(pet_feature2)
        pet_feature4 = self.pet_encode4(pet_feature3)
        pet_feature5 = self.pet_encode5(pet_feature4)
        # ct encoding
        ct_feature1 = self.ct_encode1(ct_image)
        ct_feature2 = self.ct_encode2(ct_feature1)
        ct_feature3 = self.ct_encode3(ct_feature2)
        ct_feature4 = self.ct_encode4(ct_feature3)
        ct_feature5 = self.ct_encode5(ct_feature4)
        # fusion branch
        fuse_feature5 = self.fuse5(
            torch.concat((pet_feature5, ct_feature5), dim=1))
        fuse_feature4 = self.fuse4(
            fuse_feature5, torch.concat((pet_feature4, ct_feature4), dim=1))
        fuse_feature3 = self.fuse3(
            fuse_feature4, torch.concat((pet_feature3, ct_feature3), dim=1))
        fuse_feature2 = self.fuse2(
            fuse_feature3, torch.concat((pet_feature2, ct_feature2), dim=1))
        fuse_feature1 = self.fuse1(
            fuse_feature2, torch.concat((pet_feature1, ct_feature1), dim=1))
        # pet modality_specific decoding
        pet_decode_feature4 = self.pet_decode4(
            fuse_feature5, pet_feature4)
        pet_decode_feature3 = self.pet_decode3(
            pet_decode_feature4, pet_feature3)
        pet_decode_feature2 = self.pet_decode2(
            pet_decode_feature3, pet_feature2)
        pet_decode_feature1 = self.pet_decode1(
            pet_decode_feature2, pet_feature1)
        # ct modality_specific decoding
        ct_decode_feature4 = self.ct_decode4(
            fuse_feature5, ct_feature4)
        ct_decode_feature3 = self.ct_decode3(
            ct_decode_feature4, ct_feature3)
        ct_decode_feature2 = self.ct_decode2(
            ct_decode_feature3, ct_feature2)
        ct_decode_feature1 = self.ct_decode1(
            ct_decode_feature2, ct_feature1)
        # pet segmentation output
        pet_output = self.pet_output(torch.concat(
            (pet_decode_feature1, fuse_feature1), dim=1))
        # ct segmentation output
        ct_output = self.ct_output(torch.concat(
            (ct_decode_feature1, fuse_feature1), dim=1))
        return pet_output, ct_output

class SelfAttentionXNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes  # 输出类别数
        # pet encoding
        self.pet_encode1 = DoubleConv(1, 16)
        self.pet_encode2 = Encode(16, 32)
        self.pet_encode3 = Encode(32, 64)
        self.pet_encode4 = Encode(64, 128)
        self.pet_encode5 = Encode(128, 256)
        # ct encoding
        self.ct_encode1 = DoubleConv(1, 16)
        self.ct_encode2 = Encode(16, 32)
        self.ct_encode3 = Encode(32, 64)
        self.ct_encode4 = Encode(64, 128)
        self.ct_encode5 = Encode(128, 256)
        # fusion branch
        self.fuse5 = DoubleConv(512, 256)
        self.fuse4 = FusionDecode(256, 128)
        self.fuse3 = FusionDecode(128, 64)
        self.fuse2 = FusionDecode(64, 32)
        self.fuse1 = FusionDecode(32, 16)
        # pet modality-specific decoding
        self.pet_decode4 = CrossModalityDecode(256, 128)
        self.pet_decode3 = ModalitySpecificDecode(128, 64)
        self.pet_decode2 = ModalitySpecificDecode(64, 32)
        self.pet_decode1 = ModalitySpecificDecode(32, 16)
        # ct modality-specific decoding
        self.ct_decode4 = CrossModalityDecode(256, 128)
        self.ct_decode3 = ModalitySpecificDecode(128, 64)
        self.ct_decode2 = ModalitySpecificDecode(64, 32)
        self.ct_decode1 = ModalitySpecificDecode(32, 16)
        # pet output
        self.pet_output = OutConv(32, 2)
        # ct output
        self.ct_output = OutConv(32, 2)

    def forward(self, pet_image, ct_image):
        # pet encoding
        pet_feature1 = self.pet_encode1(pet_image)
        pet_feature2 = self.pet_encode2(pet_feature1)
        pet_feature3 = self.pet_encode3(pet_feature2)
        pet_feature4 = self.pet_encode4(pet_feature3)
        pet_feature5 = self.pet_encode5(pet_feature4)
        # ct encoding
        ct_feature1 = self.ct_encode1(ct_image)
        ct_feature2 = self.ct_encode2(ct_feature1)
        ct_feature3 = self.ct_encode3(ct_feature2)
        ct_feature4 = self.ct_encode4(ct_feature3)
        ct_feature5 = self.ct_encode5(ct_feature4)
        # fusion branch
        fuse_feature5 = self.fuse5(
            torch.concat((pet_feature5, ct_feature5), dim=1))
        fuse_feature4 = self.fuse4(
            fuse_feature5, torch.concat((pet_feature4, ct_feature4), dim=1))
        fuse_feature3 = self.fuse3(
            fuse_feature4, torch.concat((pet_feature3, ct_feature3), dim=1))
        fuse_feature2 = self.fuse2(
            fuse_feature3, torch.concat((pet_feature2, ct_feature2), dim=1))
        fuse_feature1 = self.fuse1(
            fuse_feature2, torch.concat((pet_feature1, ct_feature1), dim=1))
        # pet modality_specific decoding
        pet_decode_feature4 = self.pet_decode4(
            fuse_feature5, pet_feature4, pet_feature4)
        pet_decode_feature3 = self.pet_decode3(
            pet_decode_feature4, pet_feature3)
        pet_decode_feature2 = self.pet_decode2(
            pet_decode_feature3, pet_feature2)
        pet_decode_feature1 = self.pet_decode1(
            pet_decode_feature2, pet_feature1)
        # ct modality_specific decoding
        ct_decode_feature4 = self.ct_decode4(
            fuse_feature5, ct_feature4, ct_feature4)
        ct_decode_feature3 = self.ct_decode3(
            ct_decode_feature4, ct_feature3)
        ct_decode_feature2 = self.ct_decode2(
            ct_decode_feature3, ct_feature2)
        ct_decode_feature1 = self.ct_decode1(
            ct_decode_feature2, ct_feature1)
        # pet segmentation output
        pet_output = self.pet_output(torch.concat(
            (pet_decode_feature1, fuse_feature1), dim=1))
        # ct segmentation output
        ct_output = self.ct_output(torch.concat(
            (ct_decode_feature1, fuse_feature1), dim=1))
        return pet_output, ct_output
    

class CrossAttentionXNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels  # 输入通道数
        self.n_classes = n_classes  # 输出类别数
        # pet encoding
        self.pet_encode1 = DoubleConv(1, 16)
        self.pet_encode2 = Encode(16, 32)
        self.pet_encode3 = Encode(32, 64)
        self.pet_encode4 = Encode(64, 128)
        self.pet_encode5 = Encode(128, 256)
        # ct encoding
        self.ct_encode1 = DoubleConv(1, 16)
        self.ct_encode2 = Encode(16, 32)
        self.ct_encode3 = Encode(32, 64)
        self.ct_encode4 = Encode(64, 128)
        self.ct_encode5 = Encode(128, 256)
        # fusion branch
        self.fuse5 = DoubleConv(512, 256)
        self.fuse4 = FusionDecode(256, 128)
        self.fuse3 = FusionDecode(128, 64)
        self.fuse2 = FusionDecode(64, 32)
        self.fuse1 = FusionDecode(32, 16)
        # pet modality-specific decoding
        self.pet_decode4 = CrossModalityDecode(256, 128)
        self.pet_decode3 = ModalitySpecificDecode(128, 64)
        self.pet_decode2 = ModalitySpecificDecode(64, 32)
        self.pet_decode1 = ModalitySpecificDecode(32, 16)
        # ct modality-specific decoding
        self.ct_decode4 = CrossModalityDecode(256, 128)
        self.ct_decode3 = ModalitySpecificDecode(128, 64)
        self.ct_decode2 = ModalitySpecificDecode(64, 32)
        self.ct_decode1 = ModalitySpecificDecode(32, 16)
        # pet output
        self.pet_output = OutConv(32, 2)
        # ct output
        self.ct_output = OutConv(32, 2)

    def forward(self, pet_image, ct_image):
        # pet encoding
        pet_feature1 = self.pet_encode1(pet_image)
        pet_feature2 = self.pet_encode2(pet_feature1)
        pet_feature3 = self.pet_encode3(pet_feature2)
        pet_feature4 = self.pet_encode4(pet_feature3)
        pet_feature5 = self.pet_encode5(pet_feature4)
        # ct encoding
        ct_feature1 = self.ct_encode1(ct_image)
        ct_feature2 = self.ct_encode2(ct_feature1)
        ct_feature3 = self.ct_encode3(ct_feature2)
        ct_feature4 = self.ct_encode4(ct_feature3)
        ct_feature5 = self.ct_encode5(ct_feature4)
        # fusion branch
        fuse_feature5 = self.fuse5(
            torch.concat((pet_feature5, ct_feature5), dim=1))
        fuse_feature4 = self.fuse4(
            fuse_feature5, torch.concat((pet_feature4, ct_feature4), dim=1))
        fuse_feature3 = self.fuse3(
            fuse_feature4, torch.concat((pet_feature3, ct_feature3), dim=1))
        fuse_feature2 = self.fuse2(
            fuse_feature3, torch.concat((pet_feature2, ct_feature2), dim=1))
        fuse_feature1 = self.fuse1(
            fuse_feature2, torch.concat((pet_feature1, ct_feature1), dim=1))
        # pet modality_specific decoding
        pet_decode_feature4 = self.pet_decode4(
            fuse_feature5, pet_feature4, ct_feature4)
        pet_decode_feature3 = self.pet_decode3(
            pet_decode_feature4, pet_feature3)
        pet_decode_feature2 = self.pet_decode2(
            pet_decode_feature3, pet_feature2)
        pet_decode_feature1 = self.pet_decode1(
            pet_decode_feature2, pet_feature1)
        # ct modality_specific decoding
        ct_decode_feature4 = self.ct_decode4(
            fuse_feature5, ct_feature4, pet_feature4)
        ct_decode_feature3 = self.ct_decode3(
            ct_decode_feature4, ct_feature3)
        ct_decode_feature2 = self.ct_decode2(
            ct_decode_feature3, ct_feature2)
        ct_decode_feature1 = self.ct_decode1(
            ct_decode_feature2, ct_feature1)
        # pet segmentation output
        pet_output = self.pet_output(torch.concat(
            (pet_decode_feature1, fuse_feature1), dim=1))
        # ct segmentation output
        ct_output = self.ct_output(torch.concat(
            (ct_decode_feature1, fuse_feature1), dim=1))
        return pet_output, ct_output


if __name__ == '__main__':
    device = torch.device("cuda:5")
    model = XNet(1, 2).to(device)
    pet_image = torch.randn([1, 1, 256, 256]).to(device)
    ct_image = torch.randn([1, 1, 256, 256]).to(device)
    output = model(pet_image, ct_image)
    print(output[0], output[1])
