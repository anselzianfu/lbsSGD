import argparse
from yparams import YParams

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config.yaml")
parser.add_argument('--batch_sizes', type=str)
parser.add_argument('--gpus', type=str, default='')
args = parser.parse_args()

CONFIG_FILE = args.config
CONFIG = YParams.load_all(CONFIG_FILE)
print(args.batch_sizes)
CONFIG['training'].batch_sizes = args.batch_sizes
CONFIG['general'].use_gpu = len(args.gpus) > 0
CONFIG['general'].gpus = args.gpus
