import argparse
import PIL.Image as pil_image
import numpy as np

from utils import AverageMeter, calc_psnr, calc_ssim, convert_rgb_to_y, denormalize,calc_psnr_for_eachimg,calc_ssim_for_eachimg



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_image_file', type=str, required=False,
                        default="./savedimg/Set5/4/EDSR_blanced_attention_2.png")
    #"D:/WFY/20200628SR/RDN-pytorch-master/对比图/对比图/img_001_SRF_4_SR CARN.png"
    # "D:/WFY/20200628SR/RDN-pytorch-master/对比图/对比图/img_002_SRF_4_SR (2)CARN.png"
    # "./savedimg/Set5/4/PAN_Blanced_attention_2.png"
    # "./savedimg/urban100/4/PAN_Blanced_attention_66.png"

    # "./savedimg/BSDS100/4/AWSRN_blanced_attention_87.png"
    # "D:/WFY/20200628SR/RDN-pytorch-master/PAN/B100/69015.png"
    parser.add_argument('--hr_image_file', type=str, required=False,
                        default="../classical_SR_datasets/Set5/Set5/butterfly.png")
    #"D:/WFY/20200628SR/classical_SR_datasets/Set5/Set5/butterfly.png"
    # "D:/WFY/20200628SR/classical_SR_datasets/Set14/Set14/barbara.png"
    parser.add_argument('--result_image_file', type=str, required=False,
                        default="./result/")
    opt = parser.parse_args()


    image_lr = pil_image.open(opt.lr_image_file).convert('RGB')
    image_hr = pil_image.open(opt.hr_image_file).convert('RGB')

    if (image_lr.width != image_hr.width) or (image_lr.height != image_hr.height):
        image_lr = image_lr.resize((image_hr.width,image_hr.height),resample=pil_image.BICUBIC)


    # image_lr = convert_rgb_to_y(denormalize(image_lr.squeeze(0)), dim_order='chw')
    # image_hr = convert_rgb_to_y(denormalize(image_hr.squeeze(0)), dim_order='chw')
    #
    # psnr = calc_psnr(image_hr, image_lr)
    # ssim = calc_ssim(image_hr, image_lr)


    image_hr=np.array(image_hr)
    image_lr = np.array(image_lr)
    psnr = calc_psnr_for_eachimg(image_hr, image_lr)
    ssim = calc_ssim_for_eachimg(image_hr, image_lr)
    print('PSNR SSIM: {:.5f} {:.5f}'.format(psnr, ssim))

    # output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
    # output.save(opt.result_image_file + opt.choose_net + ('{}_x{}.png'.format(filename, opt.scale)))

