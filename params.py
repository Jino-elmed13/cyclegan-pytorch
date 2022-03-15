import torch


class TrainParams():
    def __init__(self):
        # データセット関係
        self.root = './data/horse2zebra'
        self.img_type = 'jpg'
        self.phase = 'train'
        self.batch_size = 1
        self.input_nc = 3
        self.output_nc = 3
        self.size = 256
        # 前処理関係
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # 学習関係
        self.start_epoch = 0
        self.n_epochs = 50
        self.lr = 0.0002
        self.decay_epoch = 200

        self.cpu = False
        self.n_cpu = 8  # よく分からぬ
        self.device_name = "cuda:0"
        self.device = torch.device(self.device_name)
        self.load_weight = False  # 途中から学習する場合Trueにする
        self.model_path = ''  # ロードするモデルのパス
        self.generator_A2B = 'netG_A2B.pth'
        self.generator_B2A = 'netG_B2A.pth'
        self.discriminator_A = 'netD_A.pth'
        self.discriminator_B = 'netD_B.pth'

        # 損失関数の係数
        self.lambda_dis = 1
        self.lambda_adv = 1
        self.lambda_cycle = 10
        self.lambda_identity = 5


class TestParams():
    def __init__(self):
        # データセット関係
        self.root = './data/horse2zebra'
        self.img_type = 'jpg'
        self.phase = 'train'
        self.batch_size = 1
        self.input_nc = 3
        self.output_nc = 3
        self.size = 256
        # 前処理関係
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        self.cpu = False
        self.n_cpu = 8  # よく分からぬ
        self.device_name = "cuda:0"
        self.device = torch.device(self.device_name)
        self.load_weight = False
        self.model_path = 'results/0314_1542'  # ロードするモデルのパス
        self.generator_A2B = 'netG_A2B.pth'
        self.generator_B2A = 'netG_B2A.pth'
        self.cuda = True
