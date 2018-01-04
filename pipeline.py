from main import Main
from capsNet import CapsNet
from config_pipeline import cfg_1, cfg_2, cfg_3, cfg_4, cfg_5, cfg_6, cfg_7


def training_capsnet(cfg):

    CapsNet_ = CapsNet(cfg)
    Main_ = Main(CapsNet_, cfg)
    Main_.train()


def pipeline():

    # training_capsnet(cfg_1)
    # training_capsnet(cfg_2)
    training_capsnet(cfg_3)
    training_capsnet(cfg_4)
    training_capsnet(cfg_5)
    training_capsnet(cfg_6)
    training_capsnet(cfg_7)


if __name__ == '__main__':

    pipeline()
