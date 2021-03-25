import argparse
from myNet import myNet, myNet2, myNet3, myNet4, myNet4_, myNet4__, myNet5
from CARN.carn import CARN, CARN_blanced_attention
from MSRN.msrn import MSRN, MSRN_blanced_attention
from FLOPs.profile import profile
from CARN.carn import CARN, CARN_blanced_attention
from CARN.carn_m import CARN_m, CARN_m_blanced_attention
from MSRN.msrn import MSRN, MSRN_blanced_attention
from EDSR.edsr import EDSR, EDSR_blanced_attention
from AWSRN.awsrn import AWSRN, AWSRN_blanced_attention
from OISR.oisr_LF_s import oisr_LF_s, oisr_LF_s_blanced_attention
from OISR.oisr_LF_m import oisr_LF_m, oisr_LF_m_blanced_attention
from s_LWSR.s_LWSR import LWSR, LWSR_blanced_attention
from IMDN import IMDN_CAM, IMDN_SAM, IMDN_CAM_add_MAXpool, IMDN_SAM_add_AVGpool, IMDN_CBAM, IMDN_BLANCED_ATTENTION, \
    IMDN_BLANCED_ATTENTION_ADD, IMDN
from MDSR.MDSR import MDSR, MDSR_blanced_attention
from RCAN import RCAN, RCAN_blancedattention
from RCAN_ORI.RCAN import RCAN_ori, RCAN_ori_blanced_attention
from PAN.pan import PAN,PAN_Blanced_attention

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
                         "IDN or IDN_blanced_attention")
parser.add_argument('--train_file', type=str, required=False, default="./h5file_DIV2K_train_HR_x2_train")
parser.add_argument('--eval_file', type=str, required=False, default="./h5file_Set5_x2_test")
parser.add_argument('--outputs_dir', type=str, required=False, default="./checkpoint")
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
parser.add_argument('--load', type=bool, default=False)

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

width = 360
height = 240
# model = myNet4(scale=2, feature_nums=64)
model = PAN_Blanced_attention(opt)
flops, params = profile(model, input_size=(1, 3, height, width))
print('IMDN_light: {} x {}, flops: {:.10f} GFLOPs, params: {}'.format(height, width, flops / (1e9), params))


# x4
# MDSR_blanced_attention: flops: 318.2230896640 GFLOPs, params: 2865083.0
# MDSR: flops: 317.9022254080 GFLOPs, params: 2847003.0    x4:32.50 / 0.8973

# AWSRN_blanced_attention: flops: 149.9313111040 GFLOPs, params: 1587441.0
# AWSRN: flops: 149.9240529920 GFLOPs, params: 1587132.0

# s_LWSR_blanced_attention: flops: 67.1918981120 GFLOPs, params: 571310.0
# s_LWSR: flops: 67.1846400000 GFLOPs, params: 571131.0

# MSRN_blanced_attention:flops: 616.0445603840 GFLOPs, params: 6082563.0
# MSRN:flops: 615.9643443200 GFLOPs, params: 6078043.0


# IMDN_BLANCED_ATTENTION: flops: 67.7742346240 GFLOPs, params: 690126.0         +c5  flops: 72.0209674240 GFLOPs, params: 715086.0
# IMDN:flops: 71.9940075520 GFLOPs, params: 715176.0
# IMDN_CAM: flops: 71.9940075520 GFLOPs, params: 714792.0
# IMDN_SAM: flops: 71.9877898240 GFLOPs, params: 711990.0
# IMDN_CAM_add_MAXpool: flops: 72.0271851520 GFLOPs, params: 714792.0
# IMDN_SAM_add_AVGpool: flops: 72.0131850240 GFLOPs, params: 712284.0
# IMDN_BLANCED_ATTENTION_ADD: flops: 67.8328074240 GFLOPs, params: 690420.0    +c5: flops: 72.0795402240 GFLOPs, params: 715380.0
# IMDN_CBAM:flops: 72.0795402240 GFLOPs, params: 715356.0

# CARN_blanced_attention:flops: 163.1741870080 GFLOPs, params: 1593658.0
# CARN:flops: 163.1441059840 GFLOPs, params: 1591963.0

# CARN_m_blanced_attention: flops: 122.8136284160 GFLOPs, params: 1163002.0    31.98
# CARN_m: flops: 122.7835637760 GFLOPs, params: 1161307.0                      Set5:31:92 0:890    Set14:28:42 0:776     Urban100:27:44 0:730    B100:25:63 0:769

# oisr_LF_s_blanced_attention:flops: 247.3933537280 GFLOPs, params: 2112080.0
# oisr_LF_s: flops: 247.3833267200 GFLOPs, params: 2111515.0

# oisr_LF_m_blanced_attention: flops: 584.5444853760 GFLOPs, params: 4436003.0
# oisr_LF_m: flops: 584.5294120960 GFLOPs, params: 4434239.0

# EDSR_blanced_attention: flops: 190.5102684160 GFLOPs, params: 1518160.0
# EDSR: flops: 190.5002414080 GFLOPs, params: 1517595.0

# RCAN_blancedattention: flops: 1517.1431956480 GFLOPs, params: 15884779.0
# RCAN: flops: 1517.3475368960 GFLOPs, params: 15874179.0

# RCAN_ori_blanced_attention:flops: 1531.3211228160 GFLOPs, params: 15589379.0
# RCAN_ori: flops: 1530.4222310400 GFLOPs, params: 15592379.0

# MDSR_blanced_attention: flops: 318.2230896640 GFLOPs, params: 2865083.0
# MDSR: flops: 317.9022254080 GFLOPs, params: 2847003.0

# PAN:flops: 46.4334356480 GFLOPs, params: 272419.0
# PAN_Blanced_attention:flops: 44.4496855040 GFLOPs, params: 271611.0





# x3
# MDSR_blanced_attention: flops: 278.0726558720 GFLOPs, params: 2902011.0
# MDSR: flops: 277.7517916160 GFLOPs, params: 2883931.0

# AWSRN_blanced_attention: flops: 140.1190154240 GFLOPs, params: 1476456.0
# AWSRN: flops: 140.1117573120 GFLOPs, params: 1476147.0

# LWSR_blanced_attention: flops: 56.8487813120 GFLOPs, params: 580558.0
# LWSR: flops: 56.8415232000 GFLOPs, params: 580379.0

# MSRN_blanced_attention: flops: 575.8940610560 GFLOPs, params: 6119491.0
# MSRN: flops: 575.8138449920 GFLOPs, params: 6114971.0

# IMDN_BLANCED_ATTENTION: flops: 70.8579409920 GFLOPs, params: 702969.0
# IMDN:flops: flops: 70.8309811200 GFLOPs, params: 703059.0
# IMDN_CAM: flops: 70.8309811200 GFLOPs, params: 702675.0
# IMDN_SAM: flops: 70.8247633920 GFLOPs, params: 699873.0
# IMDN_CAM_add_MAXpool: flops: 70.8641587200 GFLOPs, params: 702675.0
# IMDN_SAM_add_AVGpool: flops: 70.8501585920 GFLOPs, params: 700167.0
# IMDN_BLANCED_ATTENTION_ADD: flops: 70.9165137920 GFLOPs, params: 703263.0
# IMDN_CBAM: flops: 70.9165137920 GFLOPs, params: 703239.0

# CARN_blanced_attention: flops: 122.9520486400 GFLOPs, params: 1593658.0
# CARN: flops: 122.9219676160 GFLOPs, params: 1591963.0

# CARN_m_blanced_attention: flops: 82.5914900480 GFLOPs, params: 1163002.0
# CARN_m: flops: 82.5614172160 GFLOPs, params: 1161307.0

# oisr_LF_s_blanced_attention: flops: 122.1754798080 GFLOPs, params: 1261200.0
# oisr_LF_s: flops: 122.1654528000 GFLOPs, params: 1260635.0

# oisr_LF_m_blanced_attention: flops: 440.7574200320 GFLOPs, params: 4570081.0
# oisr_LF_m: flops: 440.7423795200 GFLOPs, params: 4568317.0

# EDSR_blanced_attention: flops: 150.3598510080 GFLOPs, params: 1555088.0
# EDSR: flops: 150.3498240000 GFLOPs, params: 1554523.0

# RCAN_ori_blanced_attention: flops: 1491.1707545600 GFLOPs, params: 15626307.0
# RCAN_ori: flops: 1490.2718627840 GFLOPs, params: 15629307.0

# MDSR_blanced_attention: flops: 278.0726558720 GFLOPs, params: 2902011.0
# MDSR: flops: 277.7517916160 GFLOPs, params: 2883931.0

# PAN:flops: 35.5304407040 GFLOPs, params: 261403.0
# PAN_Blanced_attention:flops: 34.6377584640 GFLOPs, params: 260999.0






# x2
# MDSR_blanced_attention: flops: 259.5208396800 GFLOPs, params: 2717371.0
# MDSR: flops: 259.1999918080 GFLOPs, params: 2699291.0

# AWSRN_blanced_attention: flops: 133.1102760960 GFLOPs, params: 1397181.0
# AWSRN: flops: 133.1030097920 GFLOPs, params: 1396872.0

# LWSR_blanced_attention: flops: 51.9965573120 GFLOPs, params: 534318.0
# LWSR: flops: 51.9892992000 GFLOPs, params: 534139.0

# MSRN_blanced_attention: flops: 557.3423267840 GFLOPs, params: 5934851.0
# MSRN: flops: 557.2621107200 GFLOPs, params: 5930331.0

# IMDN_BLANCED_ATTENTION: flops: 70.0272066560 GFLOPs, params: 694314.0
# IMDN:flops: 70.0002467840 GFLOPs, params: 694404.0
# IMDN_CAM: flops: 70.0002467840 GFLOPs, params: 694020.0
# IMDN_SAM: flops: 69.9940290560 GFLOPs, params: 691218.0
# IMDN_CAM_add_MAXpool: flops: 70.0334243840 GFLOPs, params: 694020.0
# IMDN_SAM_add_AVGpool: flops: 70.0194242560 GFLOPs, params: 691512.0
# IMDN_BLANCED_ATTENTION_ADD: flops: 70.0857794560 GFLOPs, params: 694608.0
# IMDN_CBAM:flops: 70.0857794560 GFLOPs, params: 694584.0

# CARN_blanced_attention: flops: 104.3648184320 GFLOPs, params: 1593658.0
# CARN: flops: 104.3347374080 GFLOPs, params: 1591963.0

# CARN_m_blanced_attention: flops: 64.0042557440 GFLOPs, params: 1163002.0
# CARN_m: flops: 63.9741870080 GFLOPs, params: 1161307.0

# oisr_LF_s_blanced_attention: flops: 103.6236718080 GFLOPs, params: 1076560.0
# oisr_LF_s: flops: 103.6136448000 GFLOPs, params: 1075995.0

# oisr_LF_m_blanced_attention: flops: 374.8247306240 GFLOPs, params: 3899691.0
# oisr_LF_m: flops: 374.8096901120 GFLOPs, params: 3897927.0

# EDSR_blanced_attention: flops: 131.8080430080 GFLOPs, params: 1370448.0
# EDSR: flops: 131.7980160000 GFLOPs, params: 1369883.0

# RCAN_ori_blanced_attention: flops: 1472.6188236800 GFLOPs, params: 15441667.0
# RCAN_ori: flops: 1471.7199319040 GFLOPs, params: 15444667.0

# MDSR_blanced_attention: flops: 259.5208396800 GFLOPs, params: 2717371.0
# MDSR: flops: 259.1999918080 GFLOPs, params: 2699291.0

# PAN:flops: 28.0447488000 GFLOPs, params: 261403.0
# PAN_Blanced_attention:flops: 27.6480000000 GFLOPs, params: 260999.0