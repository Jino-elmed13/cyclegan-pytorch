# %%
# ライブラリのインポート
import glob
import os
from datetime import datetime

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import itertools
import torchsummary
from torch.utils.tensorboard import SummaryWriter

# 自作モジュールなどのインポート
from params import TrainParams
from dataset import ImageDataset, ImageTransform
from net import Generator, Discriminator

from utils import ReplayBuffer, LambdaLR
from utils import weights_init_normal


# %%
# functions
# データパスをリストで取得
def make_datapath_list(root, extention='dcm', phase='train'):
    filesA = sorted(glob.glob(os.path.join(root, '%sA' % phase) + '/*/*.' + str(extention)))
    filesB = sorted(glob.glob(os.path.join(root, '%sB' % phase) + '/*/*.' + str(extention)))
    return filesA, filesB


# loss の保存
def save_loss(train_info, batches_done):
    for k, v in train_info.items():
        writer.add_scalar(k, v, batches_done)


# %%
# パラメータ取得
param = TrainParams()
# ファイルパスリストの取得
filesA, filesB = make_datapath_list(param.root, extention=param.img_type, phase=param.phase)
# 出力フォルダ作成
outdir = os.path.join('results', datetime.now().strftime('%m%d_%H%M'))
os.makedirs(outdir, exist_ok=True)
# %%
# Log & TensorBoard
log_dir = os.path.join(outdir, 'logs')
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# 前処理，データローダーの設定
tf = ImageTransform(param.size, param.mean, param.std)
dataloader = DataLoader(
    ImageDataset(filelistA=filesA, filelistB=filesB, transform=tf, phase='train'),
    batch_size=param.batch_size, shuffle=True
)


# ネットワークの読み込み
# 生成器
netG_A2B = Generator(param.input_nc, param.output_nc)
netG_B2A = Generator(param.input_nc, param.output_nc)

# 識別器
netD_A = Discriminator(param.input_nc)
netD_B = Discriminator(param.input_nc)

# ネットワークをGPUへ
if not param.cpu:
    netG_A2B.to(param.device)
    netG_B2A.to(param.device)
    netD_A.to(param.device)
    netD_B.to(param.device)

# 重みパラメータ初期化
netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# 保存したモデルのロード
if param.load_weight is True:
    netG_A2B.load_state_dict(torch.load(os.path.join(param.model_path, param.generator_A2B), map_location=param.device_name), strict=False)
    netG_B2A.load_state_dict(torch.load(os.path.join(param.model_path, param.generator_B2A), map_location=param.device_name), strict=False)
    netD_A.load_state_dict(torch.load(os.path.join(param.model_path, param.discriminator_A), map_location=param.device_name), strict=False)
    netD_B.load_state_dict(torch.load(os.path.join(param.model_path, param.discriminator_B), map_location=param.device_name), strict=False)

# 損失関数の設定
criterion_dis = torch.nn.MSELoss()
criterion_adv = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizer & LearningRate Schedukers
optimizer_G = torch.optim.Adam(
    itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=param.lr, betas=(0.5, 0.999)
)
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=param.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=param.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(param.n_epochs, param.start_epoch, param.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(param.n_epochs, param.start_epoch, param.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(param.n_epochs, param.start_epoch, param.decay_epoch).step
)

# 入出力メモリの確保
Tensor = torch.cuda.FloatTensor if not param.cpu else torch.Tensor
input_A = Tensor(param.batch_size, param.input_nc, param.size, param.size)
input_B = Tensor(param.batch_size, param.input_nc, param.size, param.size)
target_real = Variable(Tensor(param.batch_size, 1).fill_(1.0), requires_grad=False)  # 真の画像の正解テンソル
target_fake = Variable(Tensor(param.batch_size, 1).fill_(0.0), requires_grad=False)  # 生成画像の正解テンソル

# 過去データ分のメモリ確保
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# %%
# 学習の開始
for epoch in range(param.start_epoch, param.n_epochs):
    for i, batch in enumerate(dataloader):
        # 入力の作成(真の画像)
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        # Generator
        '''
        G_A2BとGB2Aの損失計算分けたほうがいい気がする
        '''
        optimizer_G.zero_grad()  # optimizer の初期化

        # loss_identity (Identity loss)
        same_B = netG_A2B(real_B)  # G_A2B(B) = B
        loss_identity_B = criterion_identity(same_B, real_B) * param.lambda_identity

        same_A = netG_B2A(real_A)  # G_B2A(A)=A
        loss_identity_A = criterion_identity(same_A, real_A) * param.lambda_identity

        # loss_adv (Adversarial loss)
        fake_B = netG_A2B(real_A)  # real_Aをfake_Bに変換
        pred_fake = netD_B(fake_B)  # 識別器の予測
        loss_adv_A2B = criterion_adv(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_adv_B2A = criterion_adv(pred_fake, target_real)

        # loss_cycle（Cycle-consistency loss）
        recovered_A = netG_B2A(fake_B)  # A->B->A'
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * param.lambda_cycle  # expect A' == A

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * param.lambda_cycle

        # Total Generator loss
        loss_G = loss_identity_A + loss_identity_B + loss_adv_A2B + loss_adv_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()  # 誤差逆伝播
        optimizer_G.step()  # パラメーター更新

        # Discriminator_A
        optimizer_D_A.zero_grad()  # optimizer 初期化

        # 真の画像(A)に対する識別
        pred_real = netD_A(real_A)
        loss_D_real = criterion_dis(pred_real, target_real)

        # 生成画像(A)に対する識別
        fake_A = fake_A_buffer.push_and_pop(fake_A)  # 過去に生成した画像から持ってくる
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_dis(pred_fake, target_fake)

        # Total Discriminator loss
        loss_D_A = (loss_D_real + loss_D_fake) * param.lambda_dis
        loss_D_A.backward()

        # Discriminator_B
        optimizer_D_B.zero_grad()

        # 真の画像(B)に対する識別
        pred_real = netD_B(real_B)
        loss_D_real = criterion_dis(pred_real, target_real)

        # 生成画像(B)に対する識別
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_dis(pred_fake, target_fake)

        # Total Discriminator loss
        loss_D_B = (loss_D_real + loss_D_fake) * param.lambda_dis
        loss_D_B.backward()

        optimizer_D_B.step()

        if i % 20 == 0:
            print(
                'Epoch[{}]({}/{}) loss_G: {:.4f} loss_G_identity: {:.4f} loss_G_adv: {:.4f}loss_G_cycle: {:.4f} loss_D: {:.4f}'.format(
                    epoch, i, len(dataloader), loss_G, (loss_identity_A + loss_identity_B),
                    (loss_adv_A2B + loss_adv_B2A), (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B)
                )
            )
            train_info = {
                'epoch': epoch,
                'batch_num': i,
                'lossG': loss_G.item(),
                'lossG_identity': (loss_identity_A.item() + loss_identity_B.item()),
                'lossG_adv': (loss_adv_A2B.item() + loss_adv_B2A.item()),
                'lossG_cycle': (loss_cycle_ABA.item() + loss_cycle_BAB.item()),
                'lossD': (loss_D_A.item() + loss_D_B.item()),
            }

        batches_done = (epoch - 1) * len(dataloader) + i
        save_loss(train_info, batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), str(outdir) + '/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), str(outdir) + '/netG_B2A.pth')
    torch.save(netD_A.state_dict(), str(outdir) + '/netD_A.pth')
    torch.save(netD_B.state_dict(), str(outdir) + '/netD_B.pth')
# %%
# 確認用
print(filesA[0], filesB[0])
# %%
# 生成器の構造の確認
torchsummary.summary(netG_A2B, (param.input_nc, param.size, param.size))
# %%
# 識別器の構造の確認
torchsummary.summary(netD_A, (param.input_nc, param.size, param.size))
# %%
# %%
