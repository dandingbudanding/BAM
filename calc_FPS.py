import argparse
import os
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn

from IMDN import IMDN_CAM, IMDN_SAM, IMDN_CAM_add_MAXpool, IMDN_SAM_add_AVGpool, IMDN_CBAM, IMDN_BLANCED_ATTENTION, \
    IMDN_BLANCED_ATTENTION_ADD, IMDN
from DRLN.drln import DRLN, DRLN_BlancedAttention

def cals_fps(modelname, model,size):
    net = model
    time_count = 0.0
    for i in range(800):
        image = torch.randn(1, 3, size[0], size[1]).cuda()
        torch.cuda.synchronize()
        start_time = time.time()
        pred_semantic = net(image)
        torch.cuda.synchronize()
        # print(time.time() - start_time)
        if i >= 100:
            time_count = time_count + time.time() - start_time
    print(700 / time_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--choose_net', type=str, default="AWSRN_blanced_attention",
                        help="RCAN or RCAN_blancedattention or myNet or myNet2 or myNet3 or myNet4 or myNet4_8layer or myNet4_16 or myNet4_ or myNet4__ or "
                             "myNet5 or RRDBNet or RDN or MDSR or SRRESNET or SRCNN or "
                             "IMDN_CBAM or IMDN_BLANCED_ATTENTION or IMDN"  # 消融实验(Ablation experiments）
                             "IMDN_CAM or IMDN_SAM or IMDN_CAM_add_MAXpool or IMDN_SAM_add_AVGpool or IMDN_BLANCED_ATTENTION_ADD"  # 消融实验(Ablation experiments）
                             "or CARN or CARN_blanced_attention or CARN_m or CARN_m_blanced_attention"
                             "or MSRN or MSRN_blanced_attention"
                             "or EDSR or EDSR_blanced_attention"
                             "or AWSRN or AWSRN_blanced_attention"
                             "or MDSR or MDSR_blanced_attention"
                             "or oisr_LF_s or oisr_LF_s_blanced_attention or oisr_LF_m_blanced_attention"
                             "or LWSR_blanced_attention"
                             "or RCAN_ori_blanced_attention"
                             "or SAN or SAN_Blanced_Attention"
                             "or PAN or PAN_Blanced_attention"
                             "IDN or IDN_blanced_attention")
    # parser.add_argument('--eval_file', type=str, required=False, default="./h5file_Set5_x3_test")
    parser.add_argument('--outputs_dir', type=str, required=False, default="./checkpoint")
    parser.add_argument('--weights_file', type=str)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--growth_rate', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)

    # RCAN
    parser.add_argument('--num_rg', type=int, default=12)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--load', type=bool, default=True)

    # AWSRN
    # parser.add_argument('--n_resblocks', type=int, default=4,
    #                    help='number of LFB blocks')
    # parser.add_argument('--n_feats', type=int, default=32,
    #                     help='number of feature maps')
    parser.add_argument('--n_resblocks_awsrn', type=int, default=4,
                        help='number of LFB blocks')
    parser.add_argument('--n_awru_awsrn', type=int, default=4,
                        help='number of n_awru in one LFB')
    parser.add_argument('--n_feats_awsrn', type=int, default=32,
                        help='number of feature maps')
    parser.add_argument('--block_feats_awsrn', type=int, default=128,
                        help='number of feature maps')
    parser.add_argument('--res_scale_awsrn', type=float, default=1,
                        help='residual scaling')

    # EDSR
    parser.add_argument('--n_resblocks', type=int, default=16,
                        help='number of LFB blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')

    # s_LWSR
    parser.add_argument('--n_feats_s_LWSR', type=int, default=32,
                        help='number of feature maps')

    # RCAN_ORI
    parser.add_argument('--chop', action='store_true',
                        help='enable memory-efficient forward')
    # options for residual group and feature channel reduction
    parser.add_argument('--n_resgroups', type=int, default=10,
                        help='number of residual groups')
    parser.add_argument('--n_resblocks_rcan_ori', type=int, default=20,
                        help='number of residual blocks')
    parser.add_argument('--n_feats_rcan_ori', type=int, default=64,
                        help='number of feature maps')

    # oisr_LF
    parser.add_argument('--n_resblocks_oisr_LF_s', type=int, default=8,
                        help='number of residual blocks')
    parser.add_argument('--n_feats_oisr_LF_s', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--n_resblocks_oisr_LF_m', type=int, default=8,
                        help='number of residual blocks')
    parser.add_argument('--n_feats_oisr_LF_m', type=int, default=122,
                        help='number of feature maps')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')
    parser.add_argument('--act', type=str, default='prelu',
                        help='activation function')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')

    # MDSR
    # scale_list = [int(scale) for scale in opt['scale'].split(',')]

    # IDN
    parser.add_argument('--nFeat_IDN', type=int, default=64, help='number of feature maps')
    parser.add_argument('--nDiff_IDN', type=int, default=16, help='number of diff feature')
    parser.add_argument('--nFeat_slice_IDN', type=int, default=4, help='scale of slice feature')
    parser.add_argument('--patchSize_IDN', type=int, default=96, help='patch size')
    parser.add_argument('--nChannel_IDN', type=int, default=3, help='number of color channels to use')

    # EDSR
    # EDSR
    # parser.add_argument('--n_feats_edsr', type=int, default=64, help='number of feature maps')
    # parser.add_argument('--n_resblocks_edsr', type=int, default=16, help='number of diff feature')
    parser.add_argument('--n_feats_edsr', type=int, default=256, help='number of feature maps')
    parser.add_argument('--n_resblocks_edsr', type=int, default=32, help='number of diff feature')

    # SAN
    parser.add_argument('--n_resblocks_san', type=int, default=10,
                        help='number of residual blocks')
    parser.add_argument('--n_feats_san', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--n_resgroups_san', type=int, default=20,
                        help='number of residual groups')

    # PAN
    parser.add_argument('--in_nc_pan', type=int, default=3,
                        help='number of residual blocks')
    parser.add_argument('--out_nc_pan', type=int, default=3,
                        help='number of feature maps')
    parser.add_argument('--nf_pan', type=int, default=40,
                        help='number of residual groups')
    parser.add_argument('--unf_pan', type=int, default=24,
                        help='number of feature maps')
    parser.add_argument('--nb_pan', type=int, default=16,
                        help='number of residual groups')
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(opt.seed)

    scales = [2,3,4]
    # networks = ["LWSR"]
    networks = ["IMDN","IMDN_BLANCED_ATTENTION","DRLN","DRLN_BlancedAttention"]
    # "IMDN_BLANCED_ATTENTION","PAN_Blanced_attention","IMDN","PAN"
    for scale in scales:
        opt.scale = scale
        for network in networks:
            opt.choose_net = network
            # 消融实验(Ablation experiments）

            if opt.choose_net == "IMDN_BLANCED_ATTENTION":
                model = IMDN_BLANCED_ATTENTION(upscale=opt.scale).to(device)

            elif opt.choose_net == "IMDN":
                model = IMDN(upscale=opt.scale).to(device)

            elif opt.choose_net == "DRLN_BlancedAttention":
                model = DRLN_BlancedAttention(opt).to(device)
            elif opt.choose_net == "DRLN":
                model = DRLN(opt).to(device)
            print(opt.choose_net)
            for size_ in range(160,240,20):
                cals_fps(opt.choose_net, model,[size_,size_])