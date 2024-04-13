from pathlib import Path
import datetime
import os
import logging
import sys

def init_experiment_dir(cfg):
    experiment_dir = Path(cfg.log.dir)
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(
        str(experiment_dir)
        + "/Flyingthings3d-"
        + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    )
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath(cfg.ckpt.save_path)
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath("logs/")
    log_dir.mkdir(exist_ok=True)

    os.system("cp -r %s %s" % ("models", log_dir))
    os.system("cp -r %s %s" % ("dataset", log_dir))
    os.system("cp -r %s %s" % ("config", log_dir))
    os.system("cp %s %s" % ("train.py", log_dir))
    
    return checkpoints_dir, log_dir

def init_logging(log_dir, cfgs):
    logger = logging.getLogger(cfgs.model.name)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] - %(message)s")
    file_handler = logging.FileHandler(str(log_dir) + "/log_%s.log" % cfgs.model.name)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def init_log(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)