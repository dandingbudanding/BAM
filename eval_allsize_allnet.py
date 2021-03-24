import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim, convert_rgb_to_y, denormalize

from MDSR.MDSR import MDSR, MDSR_blanced_attention
from IMDN import IMDN_ACAM, IMDN_MSAM, IMDN_ACAM_add_MAXpool, IMDN_MSAM_add_AVGpool, IMDN_CBAM, IMDN_BLANCED_ATTENTION, \
    IMDN_BLANCED_ATTENTION_ADD, IMDN
from CARN.carn import CARN, CARN_blanced_attention
from CARN.carn_m import CARN_m, CARN_m_blanced_attention
from MSRN.msrn import MSRN, MSRN_blanced_attention
from EDSR.edsr import EDSR, EDSR_blanced_attention
from AWSRN.awsrn import AWSRN, AWSRN_blanced_attention
from OISR.oisr_LF_s import oisr_LF_s, oisr_LF_s_blanced_attention
from OISR.oisr_LF_m import oisr_LF_m, oisr_LF_m_blanced_attention
from s_LWSR.s_LWSR import LWSR_blanced_attention
from RCAN_ORI.RCAN import RCAN_ori_blanced_attention
from PAN.pan import PAN, PAN_Blanced_attention
from DRLN.drln import DRLN,DRLN_BlancedAttention
from torch.optim import lr_scheduler


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
    parser.add_argument('--weights_file', type=str)
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--growth_rate', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
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
    parser.add_argument('--n_feats_edsr', type=int, default=64, help='number of feature maps')
    parser.add_argument('--n_resblocks_edsr', type=int, default=16, help='number of diff feature')

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
    device = torch.device('cuda:0,1' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(opt.seed)

    scales = [2,3, 4]
    datasetfortests = ["Set5", "Set14",  "BSDS100"]
    networks = ["CARN_blanced_attention", "CARN_m_blanced_attention", "IMDN_BLANCED_ATTENTION","MSRN_blanced_attention","EDSR_blanced_attention",
               "AWSRN_blanced_attention",  "oisr_LF_s_blanced_attention","oisr_LF_m_blanced_attention", "LWSR_blanced_attention",
                "PAN_Blanced_attention", "IMDN_CAM", "IMDN_SAM", "IMDN_CAM_add_MAXpool",
               "IMDN_SAM_add_AVGpool", "IMDN_BLANCED_ATTENTION_ADD", "IMDN_CBAM"]#"RCAN_ori_blanced_attention",
    # networks=["RCAN_ori_blanced_attention"]
    for datasetfortest in datasetfortests:
        for scale in scales:
            opt.scale=scale
            for network in networks:
                opt.choose_net=network

                # 消融实验(Ablation experiments）
                if opt.choose_net == "IMDN_ACAM":
                    model = IMDN_ACAM(upscale=opt.scale).cuda()
                    criterion = nn.L1Loss()
                elif opt.choose_net == "IMDN_MSAM":
                    model = IMDN_MSAM(upscale=opt.scale).cuda()
                    criterion = nn.L1Loss()
                elif opt.choose_net == "IMDN_ACAM_add_MAXpool":
                    model = IMDN_ACAM_add_MAXpool(upscale=opt.scale).cuda()
                    criterion = nn.L1Loss()
                elif opt.choose_net == "IMDN_MSAM_add_AVGpool":
                    model = IMDN_MSAM_add_AVGpool(upscale=opt.scale).cuda()
                    criterion = nn.L1Loss()
                elif opt.choose_net == "IMDN_MSAM_add_AVGpool":
                    model = IMDN_MSAM_add_AVGpool(upscale=opt.scale).cuda()
                    criterion = nn.L1Loss()
                elif opt.choose_net == "IMDN_BLANCED_ATTENTION_ADD":
                    model = IMDN_BLANCED_ATTENTION_ADD(upscale=opt.scale).cuda()
                    criterion = nn.L1Loss()
                elif opt.choose_net == "IMDN_CBAM":
                    model = IMDN_CBAM(upscale=opt.scale).cuda()
                    criterion = nn.L1Loss()
                elif opt.choose_net == "IMDN_BLANCED_ATTENTION":
                    model = IMDN_BLANCED_ATTENTION(upscale=opt.scale).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()


                elif opt.choose_net == "IMDN":
                    model = IMDN(upscale=opt.scale).cuda()
                    criterion = nn.L1Loss()
                # 消融实验(Ablation experiments）

                elif opt.choose_net == "CARN_blanced_attention":
                    model = CARN_blanced_attention(upscale=opt.scale).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "CARN_m_blanced_attention":
                    model = CARN_m_blanced_attention(upscale=opt.scale).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "MSRN":
                    model = MSRN(upscale=opt.scale).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "MSRN_blanced_attention":
                    model = MSRN_blanced_attention(upscale=opt.scale).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "EDSR_blanced_attention":
                    model = EDSR_blanced_attention(opt).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "EDSR":
                    model = EDSR(opt).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "AWSRN_blanced_attention":
                    model = AWSRN_blanced_attention(opt).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "MDSR":
                    model = MDSR(scale=opt.scale).to(device)
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()
                elif opt.choose_net == "MDSR_blanced_attention":
                    model = MDSR_blanced_attention(scale=opt.scale).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "oisr_LF_s":
                    model = oisr_LF_s(opt).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "oisr_LF_s_blanced_attention":
                    model = oisr_LF_s_blanced_attention(opt).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "oisr_LF_m_blanced_attention":
                    model = oisr_LF_m_blanced_attention(opt).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "LWSR_blanced_attention":
                    model = LWSR_blanced_attention(opt).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()
                elif opt.choose_net == "RCAN_ori_blanced_attention":
                    model = RCAN_ori_blanced_attention(opt).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "PAN_Blanced_attention":
                    model = PAN_Blanced_attention(opt).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()

                elif opt.choose_net == "DRLN_BlancedAttention":
                    model = DRLN_BlancedAttention(opt).cuda()
                    model = torch.nn.DataParallel(model).cuda()
                    criterion = nn.L1Loss()


                if opt.load == True:
                    pth_path = './checkpoint/' + str(network) + '/x' + str(scale) + '/best.pth'
                    print('Loading weights:', pth_path)

                    checkpoint = torch.load(pth_path)
                    model.load_state_dict(checkpoint)

                    # model_dict = model.state_dict()
                    # pretrained_dict = checkpoint
                    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    # model_dict.update(pretrained_dict)
                    # model.load_state_dict(model_dict)

                eval_file_="./h5file_"+datasetfortest+"_x"+str(scale)+"_test"
                eval_dataset = EvalDataset(eval_file_)
                eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

                model.eval()
                epoch_psnr = AverageMeter()
                epoch_ssim = AverageMeter()

                for data in eval_dataloader:
                    inputs, labels = data
                    if network == "SRCNN":
                        import torch.nn.functional as F

                        inputs = F.interpolate(inputs, scale_factor=opt.scale, mode='bilinear')
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad():
                        if network == "MDSR":
                            preds = model(inputs, opt.scale)
                        elif network == "MDSR_blanced_attention":
                            preds = model(inputs, opt.scale)
                        else:
                            preds = model(inputs)

                    preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
                    labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

                    preds = preds[opt.scale:-opt.scale, opt.scale:-opt.scale]
                    labels = labels[opt.scale:-opt.scale, opt.scale:-opt.scale]

                    epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
                    epoch_ssim.update(calc_ssim(preds, labels), len(inputs))

                print('scale:{}    dataset:{}   model:{}   eval psnr: {:.6f}   ssim: {:.4f}'.format(str(scale),datasetfortest,network,epoch_psnr.avg, epoch_ssim.avg))

# python train.py --choose_net="IMDN_BLANCED_CBAM"
