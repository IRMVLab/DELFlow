import os
import yaml 
import sys
import random
import logging
import torch
import torch.optim
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
from dataset.flyingthings_subset import FlyingThings3D
from dataset.kitti import KITTI
from models import CamLiPWC
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
from utils import copy_to_device, build_optim_and_sched, FastDataLoader, init_log


class Trainer:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NCCL_IB_DISABLE"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = "1"

        self.cfgs = cfgs    
        self.curr_epoch = 1
        self.device = device
        self.n_gpus = torch.cuda.device_count()
        self.is_main = device.index is None or device.index == 0
        self.cfgs.log.dir = (
            self.cfgs.log.dir
            + "/{}/".format(self.cfgs.model.name)
            + str(datetime.now().strftime("%Y-%m-%d_%H-%M"))
        )
        os.makedirs(self.cfgs.log.dir, exist_ok=True)
        init_log(os.path.join(self.cfgs.log.dir, "train.log"))

        if device.index is None:
            logging.info("No CUDA device detected, using CPU for training")
        else:
            logging.info(
                "Using GPU %d: %s" % (device.index, torch.cuda.get_device_name(device))
            )
            logging.info(
                "PID:{}".format(os.getpid()) 
            )
            if self.n_gpus > 1:
                init_process_group(
                    "nccl",
                    "tcp://localhost:%d" % self.cfgs.port,
                    world_size=self.n_gpus,
                    rank=self.device.index,
                )
                self.cfgs.model.batch_size = int(
                    self.cfgs.model.batch_size / self.n_gpus
                )
                self.cfgs.trainset.n_workers = int(
                    self.cfgs.trainset.n_workers / self.n_gpus
                )
                self.cfgs.valset.n_workers = int(
                    self.cfgs.valset.n_workers / self.n_gpus
                )
              
            cudnn.benchmark = False
            torch.cuda.set_device(self.device)

        if self.is_main:
            logging.info("Logs will be saved to %s" % self.cfgs.log.dir)
            self.summary_writer = SummaryWriter(self.cfgs.log.dir)
            logging.info("Configurations:\n" + OmegaConf.to_yaml(self.cfgs))
            os.system("cp -r %s %s" % ("models", self.cfgs.log.dir))
            os.system("cp -r %s %s" % ("dataset", self.cfgs.log.dir))
            os.system("cp -r %s %s" % ("config", self.cfgs.log.dir))
            os.system("cp %s %s" % ("train.py", self.cfgs.log.dir))
        else:
            logging.root.disabled = True
        
        if self.cfgs.trainset.name == "flyingthings3d":
            self.train_dataset = FlyingThings3D(self.cfgs.trainset)      
            self.val_dataset = FlyingThings3D(self.cfgs.valset)
        elif self.cfgs.trainset.name == "kitti":
            self.train_dataset = KITTI(self.cfgs.trainset)
            self.val_dataset = KITTI(self.cfgs.valset)
        else:
            raise NotImplementedError
        
        logging.info("Loading training set from %s" % self.cfgs.trainset.root_dir)
        self.train_sampler = (
            DistributedSampler(self.train_dataset) if self.n_gpus > 1 else None
        )
        logging.info("Loading validation set from %s" % self.cfgs.valset.root_dir)
        self.val_sampler = (
            DistributedSampler(self.val_dataset) if self.n_gpus > 1 else None
        )

        self.train_loader = FastDataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.cfgs.trainset.n_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=self.cfgs.trainset.drop_last,
        )

        self.val_loader = FastDataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfgs.model.batch_size,
            shuffle=False,
            num_workers=self.cfgs.valset.n_workers,
            pin_memory=True,
            sampler=self.val_sampler,
        )

        logging.info("Creating model: %s" % self.cfgs.model.name)
        
        self.model = CamLiPWC(self.cfgs.model)
        self.model.to(device=self.device)

        n_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        logging.info("Trainable parameters: %d (%.1fM)" % (n_params, n_params / 1e6))

        if self.n_gpus > 1:
            if self.cfgs.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.ddp = DistributedDataParallel(self.model, [self.device.index])
        else:
            self.ddp = self.model

        self.best_metrics = None
        if self.cfgs.ckpt.path is not None:
            self.load_ckpt(self.cfgs.ckpt.path, resume=self.cfgs.ckpt.resume)

        logging.info("Creating optimizer: %s" % self.cfgs.training.opt)
        self.optimizer, self.scheduler = build_optim_and_sched(
            self.cfgs.training, self.model
        )
        self.scheduler.step(self.curr_epoch - 1)

        self.amp_scaler = GradScaler(enabled=self.cfgs.amp)

    def run(self):
        while self.curr_epoch <= self.cfgs.training.epochs:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.curr_epoch)
            if self.val_sampler is not None:
                self.val_sampler.set_epoch(self.curr_epoch)

            self.train_one_epoch()

            if self.curr_epoch % self.cfgs.val_interval == 0:
                self.validate()

            self.save_ckpt()
            self.scheduler.step(self.curr_epoch)

            self.curr_epoch += 1

    def train_one_epoch(self):
        logging.info("Start training...")

        self.ddp.train()
        self.model.clear_metrics()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]["lr"]
        self.save_scalar_summary({"learning_rate": lr}, prefix="train")

        logging.info("Epoch: [%d/%d]" % (self.curr_epoch, self.cfgs.training.epochs))
        logging.info("Current learning rate: %.8f" % lr)

        for i, inputs in enumerate(self.train_loader):
            inputs = copy_to_device(inputs, self.device)

            # forward
            with torch.cuda.amp.autocast(enabled=self.cfgs.amp):
                self.ddp.forward(inputs)
                loss = self.model.get_loss()

            # backward
            self.amp_scaler.scale(loss).backward()

            # grad clip
            if "grad_max_norm" in self.cfgs.training.keys():
                self.amp_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(),
                    max_norm=self.cfgs.training.grad_max_norm,
                )

            # update
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
            self.optimizer.zero_grad()

        metrics = self.model.get_metrics()
        self.save_scalar_summary(metrics, prefix="train")

    @torch.no_grad()
    def validate(self):
        logging.info("Start validating...")

        self.ddp.eval()
        self.model.clear_metrics()

        for i, inputs in enumerate(self.val_loader):
            inputs = copy_to_device(inputs, self.device)
            
            with torch.cuda.amp.autocast(enabled=False):
                self.ddp.forward(inputs)

        metrics = self.model.get_metrics()
        self.save_scalar_summary(metrics, prefix="val")

        for k, v in metrics.items():
            logging.info("%s: %.4f" % (k, v))

        if self.model.is_better(metrics, self.best_metrics):
            self.best_metrics = metrics
            self.save_ckpt("best.pt")

    def save_scalar_summary(self, scalar_summary: dict, prefix):
        if self.is_main and self.cfgs.log.save_scalar_summary:
            for name in scalar_summary.keys():
                self.summary_writer.add_scalar(
                    prefix + "/" + name, scalar_summary[name], self.curr_epoch
                )

    def save_ckpt(self, filename=None):
        if self.is_main and self.cfgs.log.save_ckpt:
            ckpt_dir = os.path.join(self.cfgs.log.dir, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)
            # filepath = os.path.join(
            #     ckpt_dir, filename or "epoch-%03d.pt" % self.curr_epoch
            # )
            filepath = os.path.join(
                ckpt_dir, filename or "epoch-latest.pt"
            )
            logging.info("Saving checkpoint to %s" % filepath)
            torch.save(
                {
                    "last_epoch": self.curr_epoch,
                    "state_dict": self.model.state_dict(),
                    "best_metrics": self.best_metrics,
                },
                filepath,
            )

    def load_ckpt(self, filepath, resume=True):
        logging.info("Loading checkpoint from %s" % filepath)
        checkpoint = torch.load(filepath, self.device)
        if resume:
            self.curr_epoch = checkpoint["last_epoch"] + 1
            self.best_metrics = checkpoint["best_metrics"]
            logging.info("Current best metrics: %s" % str(self.best_metrics))
        # self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.load_state_dict(checkpoint["state_dict"], strict=False)


def create_trainer(device_id, cfgs):
    device = torch.device("cpu" if device_id is None else "cuda:%d" % device_id)
    trainer = Trainer(device, cfgs)
    trainer.run()


def main(cfgs: DictConfig):
    # set num_workers of data loader
    if not cfgs.debug:
        n_devices = max(torch.cuda.device_count(), 1)
        cfgs.trainset.n_workers = min(
            os.cpu_count(), cfgs.trainset.n_workers * n_devices
        )
        cfgs.valset.n_workers = min(os.cpu_count(), cfgs.valset.n_workers * n_devices)
    else:
        cfgs.trainset.n_workers = 0
        cfgs.valset.n_workers = 0

    if cfgs.port == "random":
        cfgs.port = random.randint(10000, 20000)

    if cfgs.training.accum_iter > 1:
        cfgs.model.batch_size //= int(cfgs.training.accum_iter)

    # create trainers
    if torch.cuda.device_count() == 0:  # CPU
        create_trainer(None, cfgs)
    elif torch.cuda.device_count() == 1:  # Single GPU
        create_trainer(0, cfgs)
    elif torch.cuda.device_count() > 1:  # Multiple GPUs
        mp.spawn(create_trainer, (cfgs,), torch.cuda.device_count())


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))
    main(cfgs)
