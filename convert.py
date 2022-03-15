# %%
# ライブラリのインポート
import glob
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ipywidgets import interact
# import numpy as np
# from PIL import Image
# 自作モジュールのインポート
from params import TestParams
from dataset import ImageDataset, ImageTransform
from net import Generator


# %%
# functions
# データパスをリスト形式で取得
def make_datapath_list(root, extention='dcm', phase='train'):
    filesA = sorted(glob.glob(os.path.join(root, '%sA' % phase) + '/*/*.' + str(extention)))
    filesB = sorted(glob.glob(os.path.join(root, '%sB' % phase) + '/*/*.' + str(extention)))
    if len(filesA) == 0 or len(filesB) == 0:
        print('Cannot find files')
    return filesA, filesB


# %%
# パラメータ取得
param = TestParams()
# ファイルパスリスト取得
filesA, filesB = make_datapath_list(param.root, param.img_type, param.phase)

# 出力フォルダの作成
model_path = param.model_path  # ロードするモデルのディレクトリ
outdir = os.path.join(model_path, 'output')
os.makedirs(outdir, exist_ok=True)

# %%
# 前処理，データローダーの設定
tf = ImageTransform(param.size, param.mean, param.std)
dataloader = DataLoader(
    ImageDataset(filelistA=filesA, filelistB=filesB, transform=tf, phase='test'),
    batch_size=param.batch_size, shuffle=False
)

# ネットワークの読み込み
netG_A2B = Generator(param.input_nc, param.output_nc)
netG_B2A = Generator(param.input_nc, param.output_nc)

# CUDA
if param.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load model
netG_A2B.load_state_dict(torch.load(os.path.join(param.model_path, param.generator_A2B)))
netG_B2A.load_state_dict(torch.load(os.path.join(param.model_path, param.generator_B2A)))

# モデルをテストモードに設定
netG_A2B.eval()
netG_B2A.eval()

# 入力のメモリ確保
Tensor = torch.cuda.FloatTensor if param.cuda else torch.Tensor
input_A = Tensor(param.batch_size, param.input_nc, param.size, param.size)
input_B = Tensor(param.batch_size, param.input_nc, param.size, param.size)

# %%
real_A_img = []
real_B_img = []
fake_A_img = []
fake_B_img = []

for i, batch in enumerate(dataloader):

    # input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # generate
    fake_B = netG_A2B(real_A)
    fake_A = netG_B2A(real_B)

    real_A_out = (0.5 * (real_A.data + 1.0)).to('cpu').detach().numpy()[0, :, :, :].transpose((1, 2, 0))
    real_B_out = (0.5 * (real_B.data + 1.0)).to('cpu').detach().numpy()[0, :, :, :].transpose((1, 2, 0))

    fake_A_out = (0.5 * (fake_A.data + 1.0)).to('cpu').detach().numpy()[0, :, :, :].transpose((1, 2, 0))
    fake_B_out = (0.5 * (fake_B.data + 1.0)).to('cpu').detach().numpy()[0, :, :, :].transpose((1, 2, 0))

    real_A_img.append(real_A_out)
    real_B_img.append(real_B_out)

    fake_A_img.append(fake_A_out)
    fake_B_img.append(fake_B_out)

    if i > 100:
        break


# %%
def f(k):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
    a1 = axes[0]
    a2 = axes[1]
    a1.imshow(real_B_img[k])
    a2.imshow(fake_A_img[k])
    a1.set_title('Original')
    a2.set_title('Fake')


interact(f, k=(0, len(real_A_img)))
# %%
