import logging
import os
import sys

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import datetime
import random

import numpy as np
import torch
from torch import optim

import trainer as Trainer
from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import losses, networks, optims
from utils.configs import get_options
from utils.torch.callbacks import CheckpointsCallback

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(cfg: Config):
    logging.info("Initializing model...")
    # Model
    try:
        network = getattr(networks, cfg.model_type)(cfg)
        network.to(device)

        # === ĐẶT ĐOẠN MÃ ĐỂ XEM TỔNG THAM SỐ TẠI ĐÂY ===
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

        logging.info(f"Tổng số tham số của mô hình: {total_params:,}")
        logging.info(f"Tổng số tham số có thể huấn luyện: {trainable_params:,}")
        # === KẾT THÚC ĐOẠN MÃ XEM TỔNG THAM SỐ ===

    except AttributeError:
        raise NotImplementedError("Model {} is not implemented".format(cfg.model_type))

    logging.info("Initializing checkpoint directory and dataset...")
    # Preapre the checkpoint directory
    cfg.checkpoint_dir = checkpoint_dir = os.path.join(
        os.path.abspath(cfg.checkpoint_dir),
        cfg.name,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    log_dir = os.path.join(checkpoint_dir, "logs")
    weight_dir = os.path.join(checkpoint_dir, "weights")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    cfg.save(cfg)

    # Vùng CẦN THAY THẾ trong train.py
    try:
        # Lấy class loss từ cfg.loss_type
        LossClass = getattr(losses, cfg.loss_type)
        
        # Lấy các giá trị từ cfg. Đảm bảo các thuộc tính này có trong lớp Config của bạn.
        # Bạn đã xác nhận chúng có rồi, nên phần này sẽ hoạt động.
        ce_weight = cfg.ce_weight
        contrastive_weight = cfg.contrastive_weight
        temperature = cfg.temperature

        # Khởi tạo criterion với các đối số riêng biệt
        if cfg.loss_type == "BYOLLoss":
            criterion = LossClass(
                temperature=temperature
            )
        elif cfg.loss_type == "CEContrastiveLoss":  # ví dụ
            criterion = LossClass(
                ce_weight=ce_weight,
                contrastive_weight=contrastive_weight,
                temperature=temperature
            )
        else:
            criterion = LossClass()
        criterion.to(device)
    except AttributeError:
        raise NotImplementedError("Loss {} is not implemented".format(cfg.loss_type))
    # Kết thúc vùng CẦN THAY THẾ


    try:
        trainer = getattr(Trainer, cfg.trainer)(
            cfg=cfg,
            network=network,
            criterion=criterion,
            log_dir=cfg.checkpoint_dir,
        )
    except AttributeError:
        raise NotImplementedError("Trainer {} is not implemented".format(cfg.trainer))

    train_ds, test_ds = build_train_test_dataset(cfg)
    logging.info("Initializing trainer...")

    logging.info("Start training...")

    optimizer = optims.get_optim(cfg, network)
    lr_scheduler = None
    if cfg.learning_rate_step_size is not None:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.learning_rate_step_size,
            gamma=cfg.learning_rate_gamma,
        )

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=weight_dir,
        save_freq=cfg.save_freq,
        max_to_keep=cfg.max_to_keep,
        save_best_val=cfg.save_best_val,
        save_all_states=cfg.save_all_states,
    )

    if cfg.resume:
        trainer.load_all_states(cfg.resume_path)

    trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
    trainer.fit(train_ds, cfg.num_epochs, test_ds, callbacks=[ckpt_callback])


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="./anhnv/research/RAFM-MER/RAFM_SER/src/configs/base.py")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = get_options(args.config)
    if cfg.resume and cfg.cfg_path is not None:
        resume = cfg.resume
        resume_path = cfg.resume_path
        cfg.load(cfg.cfg_path)
        cfg.resume = resume
        cfg.resume_path = resume_path

    main(cfg)
# import logging
# import os
# import sys

# lib_path = os.path.abspath("").replace("scripts", "src")
# sys.path.append(lib_path)

# import argparse
# import datetime
# import random

# import numpy as np
# import torch
# from torch import optim

# import trainer as Trainer
# from configs.base import Config
# from data.dataloader import build_train_test_dataset
# from models import losses, networks, optims
# from utils.configs import get_options
# from utils.torch.callbacks import CheckpointsCallback

# SEED = 0
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def main(cfg: Config):
#     logging.info("Initializing model...")
#     # Model
#     try:
#         network = getattr(networks, cfg.model_type)(cfg)
#         network.to(device)
#     except AttributeError:
#         raise NotImplementedError("Model {} is not implemented".format(cfg.model_type))

#     logging.info("Initializing checkpoint directory and dataset...")
#     # Preapre the checkpoint directory
#     cfg.checkpoint_dir = checkpoint_dir = os.path.join(
#         os.path.abspath(cfg.checkpoint_dir),
#         cfg.name,
#         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
#     )
#     log_dir = os.path.join(checkpoint_dir, "logs")
#     weight_dir = os.path.join(checkpoint_dir, "weights")
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(weight_dir, exist_ok=True)
#     cfg.save(cfg)

#     # Vùng CẦN THAY THẾ trong train.py
#     try:
#         # Lấy class loss từ cfg.loss_type
#         LossClass = getattr(losses, cfg.loss_type)
        
#         # Lấy các giá trị từ cfg. Đảm bảo các thuộc tính này có trong lớp Config của bạn.
#         # Bạn đã xác nhận chúng có rồi, nên phần này sẽ hoạt động.
#         ce_weight = cfg.ce_weight
#         contrastive_weight = cfg.contrastive_weight
#         temperature = cfg.temperature

#         # Khởi tạo criterion với các đối số riêng biệt
#         criterion = LossClass(
#             ce_weight=ce_weight,
#             contrastive_weight=contrastive_weight,
#             temperature=temperature
#         )
#         criterion.to(device)
#     except AttributeError:
#         raise NotImplementedError("Loss {} is not implemented".format(cfg.loss_type))
#     # Kết thúc vùng CẦN THAY THẾ


#     try:
#         trainer = getattr(Trainer, cfg.trainer)(
#             cfg=cfg,
#             network=network,
#             criterion=criterion,
#             log_dir=cfg.checkpoint_dir,
#         )
#     except AttributeError:
#         raise NotImplementedError("Trainer {} is not implemented".format(cfg.trainer))

#     train_ds, test_ds = build_train_test_dataset(cfg)
#     logging.info("Initializing trainer...")

#     logging.info("Start training...")

#     optimizer = optims.get_optim(cfg, network)
#     lr_scheduler = None
#     if cfg.learning_rate_step_size is not None:
#         lr_scheduler = optim.lr_scheduler.StepLR(
#             optimizer,
#             step_size=cfg.learning_rate_step_size,
#             gamma=cfg.learning_rate_gamma,
#         )

#     ckpt_callback = CheckpointsCallback(
#         checkpoint_dir=weight_dir,
#         save_freq=cfg.save_freq,
#         max_to_keep=cfg.max_to_keep,
#         save_best_val=cfg.save_best_val,
#         save_all_states=cfg.save_all_states,
#     )

#     if cfg.resume:
#         trainer.load_all_states(cfg.resume_path)

#     trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
#     trainer.fit(train_ds, cfg.num_epochs, test_ds, callbacks=[ckpt_callback])


# def arg_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-cfg", "--config", type=str, default="./anhnv/research/RAFM-MER/RAFM_SER/src/configs/base.py")
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = arg_parser()
#     cfg: Config = get_options(args.config)
#     if cfg.resume and cfg.cfg_path is not None:
#         resume = cfg.resume
#         resume_path = cfg.resume_path
#         cfg.load(cfg.cfg_path)
#         cfg.resume = resume
#         cfg.resume_path = resume_path

#     main(cfg)