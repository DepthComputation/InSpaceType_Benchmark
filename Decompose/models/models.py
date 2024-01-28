from models.upsample import Up, UpSampleBN, CALayer
from models.ViT import ViT

import torch.nn as nn
import torch.nn.functional

from efficientnet_pytorch import EfficientNet


def create_model(model_name, MDR=False):
    # Create model
    if model_name == 'depth_effB5':
        model = Model_depth_effB5(MDR=MDR)

    if model_name == 'depth_decomp_effB5':
        model = Model_depth_decomp_effB5(MDR=MDR)

    return model


class Model_depth_effB5(nn.Module):
    def __init__(self, MDR=False):
        super(Model_depth_effB5, self).__init__()
        self.encoder = EfficientNet.from_pretrained(model_name='efficientnet-b5')
        self.MDR = MDR
        num_channels_d32_in = 2048
        num_channels_d32_out = 1024

        self.conv_enc_CA_1_m = CALayer(channel=512, reduction=16)
        self.conv_enc_CA_2_m = CALayer(channel=176, reduction=16)
        self.conv_enc_CA_3_m = CALayer(channel=64, reduction=16)
        self.conv_enc_CA_4_m = CALayer(channel=40, reduction=8)

        self.conv_m_CA = CALayer(channel=num_channels_d32_in, reduction=16)
        self.conv_m32 = nn.Conv2d(num_channels_d32_in, num_channels_d32_out, kernel_size=1, stride=1)

        self.up1_m = UpSampleBN(skip_input=num_channels_d32_out // 1 + 512, output_features=num_channels_d32_out // 2)
        self.up2_m = UpSampleBN(skip_input=num_channels_d32_out // 2 + 176, output_features=num_channels_d32_out // 4)
        self.up3_m = UpSampleBN(skip_input=num_channels_d32_out // 4 + 64, output_features=num_channels_d32_out // 8)
        self.up4_m = UpSampleBN(skip_input=num_channels_d32_out // 8 + 40, output_features=num_channels_d32_out // 16)
        self.up5_m = Up(num_input_channels=num_channels_d32_out // 16, num_output_channels=num_channels_d32_out // 32)

        self.conv_metric = nn.Conv2d(num_channels_d32_out // 32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        encoder_features = self.encoder.extract_endpoints(x)

        metric_depth_features = self.conv_m_CA(encoder_features['reduction_6'])
        metric_depth_features = self.conv_m32(metric_depth_features)

        up1_m = self.up1_m(metric_depth_features, self.conv_enc_CA_1_m(encoder_features['reduction_5']))
        up2_m = self.up2_m(up1_m, self.conv_enc_CA_2_m(encoder_features['reduction_4']))
        up3_m = self.up3_m(up2_m, self.conv_enc_CA_3_m(encoder_features['reduction_3']))
        up4_m = self.up4_m(up3_m, self.conv_enc_CA_4_m(encoder_features['reduction_2']))
        up5_m = self.up5_m(up4_m)
        metric_depth = self.conv_metric(up5_m)

        return metric_depth


class Model_depth_decomp_effB5(nn.Module):
    def __init__(self, MDR=False):
        super(Model_depth_decomp_effB5, self).__init__()
        self.encoder = EfficientNet.from_pretrained(model_name='efficientnet-b5')
        self.MDR = MDR
        num_channels_d32_in = 2048
        num_channels_d32_out = 1024

        self.conv_enc_CA_1_g = CALayer(channel=512, reduction=16)
        self.conv_enc_CA_2_g = CALayer(channel=176, reduction=16)
        self.conv_enc_CA_3_g = CALayer(channel=64, reduction=16)
        self.conv_enc_CA_4_g = CALayer(channel=40, reduction=8)

        self.conv_g_CA = CALayer(channel=num_channels_d32_in, reduction=16)
        self.conv_g32 = nn.Conv2d(num_channels_d32_in, num_channels_d32_out, kernel_size=1, stride=1)

        self.up1_g = UpSampleBN(skip_input=num_channels_d32_out // 1 + 512, output_features=num_channels_d32_out // 2)
        self.up2_g = UpSampleBN(skip_input=num_channels_d32_out // 2 + 176, output_features=num_channels_d32_out // 4)
        self.up3_g = UpSampleBN(skip_input=num_channels_d32_out // 4 + 64, output_features=num_channels_d32_out // 8)
        self.up4_g = UpSampleBN(skip_input=num_channels_d32_out // 8 + 40, output_features=num_channels_d32_out // 16)
        self.up5_g = Up(num_input_channels=num_channels_d32_out // 16, num_output_channels=num_channels_d32_out // 32)

        self.conv_grad = nn.Conv2d(num_channels_d32_out // 32, 2, kernel_size=3, stride=1, padding=1)

        self.conv_enc_CA_1_n = CALayer(channel=512, reduction=16)
        self.conv_enc_CA_2_n = CALayer(channel=176, reduction=16)
        self.conv_enc_CA_3_n = CALayer(channel=64, reduction=16)
        self.conv_enc_CA_4_n = CALayer(channel=40, reduction=8)

        self.conv_n_CA = CALayer(channel=num_channels_d32_in, reduction=16)
        self.conv_n32 = nn.Conv2d(num_channels_d32_in, num_channels_d32_out, kernel_size=1, stride=1)

        self.up1_n = UpSampleBN(skip_input=num_channels_d32_out // 1 + 512, output_features=num_channels_d32_out // 2)
        self.up2_n = UpSampleBN(skip_input=num_channels_d32_out // 2 + 176, output_features=num_channels_d32_out // 4)
        self.up3_n = UpSampleBN(skip_input=num_channels_d32_out // 4 + 64, output_features=num_channels_d32_out // 8)
        self.up4_n = UpSampleBN(skip_input=num_channels_d32_out // 8 + 40, output_features=num_channels_d32_out // 16)
        self.up5_n = Up(num_input_channels=num_channels_d32_out // 16, num_output_channels=num_channels_d32_out // 32)

        self.conv_norm = nn.Conv2d(num_channels_d32_out // 32, 1, kernel_size=3, stride=1, padding=1)

        self.conv_enc_CA_1_m = CALayer(channel=512, reduction=16)
        self.conv_enc_CA_2_m = CALayer(channel=176, reduction=16)
        self.conv_enc_CA_3_m = CALayer(channel=64, reduction=16)
        self.conv_enc_CA_4_m = CALayer(channel=40, reduction=8)

        self.conv_m_CA = CALayer(channel=num_channels_d32_in, reduction=16)
        self.conv_m32 = nn.Conv2d(num_channels_d32_in, num_channels_d32_out, kernel_size=1, stride=1)

        self.up1_m = UpSampleBN(skip_input=num_channels_d32_out // 1 + 512, output_features=num_channels_d32_out // 2)
        self.up2_m = UpSampleBN(skip_input=num_channels_d32_out // 2 + 176, output_features=num_channels_d32_out // 4)
        self.up3_m = UpSampleBN(skip_input=num_channels_d32_out // 4 + 64, output_features=num_channels_d32_out // 8)
        self.up4_m = UpSampleBN(skip_input=num_channels_d32_out // 8 + 40, output_features=num_channels_d32_out // 16)
        self.up5_m = Up(num_input_channels=num_channels_d32_out // 16, num_output_channels=num_channels_d32_out // 32)

        self.conv_metric = nn.Conv2d(num_channels_d32_out // 32, 1, kernel_size=3, stride=1, padding=1)
        ################################################################################################
        if self.MDR:
            self.VIT = ViT(in_channels=num_channels_d32_out // 16,
                           n_query_channels=191,
                           patch_size=8,
                           regression_output_channels=1,
                           embedding_dim=128)
            self.conv_ViT = nn.Conv2d(191, num_channels_d32_out // 16, kernel_size=1, stride=1)

    def forward(self, x):
        encoder_features = self.encoder.extract_endpoints(x)
        ################################################################################################
        grad_depth_features = self.conv_g_CA(encoder_features['reduction_6'])
        grad_depth_features = self.conv_g32(grad_depth_features)

        up1_g = self.up1_g(grad_depth_features, self.conv_enc_CA_1_g(encoder_features['reduction_5']))
        up2_g = self.up2_g(up1_g, self.conv_enc_CA_2_g(encoder_features['reduction_4']))
        up3_g = self.up3_g(up2_g, self.conv_enc_CA_3_g(encoder_features['reduction_3']))
        up4_g = self.up4_g(up3_g, self.conv_enc_CA_4_g(encoder_features['reduction_2']))
        up5_g = self.up5_g(up4_g)

        grad_depth = self.conv_grad(up5_g)
        ################################################################################################
        norm_depth_features = self.conv_n_CA(encoder_features['reduction_6'])
        norm_depth_features = self.conv_n32(norm_depth_features)

        up1_n = self.up1_n(grad_depth_features + norm_depth_features, self.conv_enc_CA_1_n(encoder_features['reduction_5']))
        up2_n = self.up2_n(up1_g + up1_n, self.conv_enc_CA_2_n(encoder_features['reduction_4']))
        up3_n = self.up3_n(up2_g + up2_n, self.conv_enc_CA_3_n(encoder_features['reduction_3']))
        up4_n = self.up4_n(up3_g + up3_n, self.conv_enc_CA_4_n(encoder_features['reduction_2']))
        up5_n = self.up5_n(up4_n)

        norm_depth = self.conv_norm(up5_n)
        ################################################################################################
        metric_depth_features = self.conv_m_CA(encoder_features['reduction_6'])
        metric_depth_features = self.conv_m32(metric_depth_features)

        up1_m = self.up1_m(norm_depth_features + metric_depth_features, self.conv_enc_CA_1_m(encoder_features['reduction_5']))
        up2_m = self.up2_m(up1_n + up1_m, self.conv_enc_CA_2_m(encoder_features['reduction_4']))
        up3_m = self.up3_m(up2_n + up2_m, self.conv_enc_CA_3_m(encoder_features['reduction_3']))
        up4_m = self.up4_m(up3_n + up3_m, self.conv_enc_CA_4_m(encoder_features['reduction_2']))

        if self.MDR:
            regression_head, attention_maps = self.VIT(up4_m)
            up4_m = self.conv_ViT(attention_maps)
            up5_m = self.up5_m(up4_m)
            metric_depth = self.conv_metric(up5_m)
            metric_depth += regression_head.unsqueeze(-1).unsqueeze(-1)

        else:
            up5_m = self.up5_m(up4_m)
            metric_depth = self.conv_metric(up5_m)

        return metric_depth
