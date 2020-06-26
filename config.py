import os
import time
import random
import yaml
from easydict import EasyDict
import argparse

def load_config(config_file):
    with open(config_file, "r") as fin:
        cfg = EasyDict(yaml.safe_load(fin))

    return cfg

def create_directory(cfg):
    keywords = [cfg.evaluation.task]
    # keywords.append("thres_"+str(cfg.truncation))
    exp_dir = "/".join(keywords)
    output_dir = os.path.join(cfg.output_dir, exp_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))
    file_made = False
    while not file_made:
        try:
            os.makedirs(output_dir)
            file_made = True
        except FileExistsError:
            timeDelay = random.randrange(1, 10)
            time.sleep(timeDelay)
            output_dir = os.path.join(cfg.output_dir, exp_dir,
                                          time.strftime("%Y-%m-%d-%H-%M-%S"))
    cfg.output_dir = output_dir
    os.chdir(cfg.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config')
    pa = parser.parse_args()
    args = load_config(pa.config)
    create_directory(args)
    print(os.getcwd())
