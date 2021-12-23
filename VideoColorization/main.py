import argparse
import matplotlib.pyplot as plt
from model_files import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path',
                    default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true',
                    help='whether to use GPU')
parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                    help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
    colorizer_siggraph17.cuda()
path=opt.img_path
# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
if(opt.use_gpu):
    tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat(
    (0*tens_l_orig, 0*tens_l_orig), dim=1))
out_img_siggraph17 = postprocess_tens(
    tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

plt.imsave('%s%s_siggraph.png' % (opt.save_prefix,path[23:-4]), out_img_siggraph17)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')
plt.show()
