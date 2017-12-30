from config import cfg

for key in cfg.keys():
    print('{}: {}'.format(key, cfg[key]))
