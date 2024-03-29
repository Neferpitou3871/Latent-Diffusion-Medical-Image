# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging

import os
import sys
from pathlib import Path

import torch
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP


from utils import KL_loss, define_instance, prepare_dataloader, setup_ddp
#from visualize_image import visualize_one_slice_in_3d_image


def main():
    parser = argparse.ArgumentParser(description="latent diffusion model")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

    # Step 0: configuration
    
    
    rank = 0
    world_size = 1
    device = 0

    torch.cuda.set_device(device)
    print(f"Using {device}")

    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["num_channels"]) - 1)
    train_loader, val_loader = prepare_dataloader(
        args,
        args.autoencoder_train["batch_size"],
        args.autoencoder_train["patch_size"],
        randcrop=True,
        rank=rank,
        world_size=world_size,
        cache=1.0,
        download=False,
        size_divisible=size_divisible,
        amp=False,
    )

    # Step 2: Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        num_channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)
    

    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
    trained_d_path = os.path.join(args.model_dir, "discriminator.pt")
    trained_g_path_last = os.path.join(args.model_dir, "autoencoder_last.pt")
    trained_d_path_last = os.path.join(args.model_dir, "discriminator_last.pt")

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if args.resume_ckpt: #if we wanted to resume from a certain checkpoint
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
            print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")
        except:
            print(f"Rank {rank}: Train autoencoder from scratch.")

        try:
            discriminator.load_state_dict(torch.load(trained_d_path, map_location=map_location))
            print(f"Rank {rank}: Load trained discriminator from {trained_d_path}")
        except:
            print(f"Rank {rank}: Train discriminator from scratch.")

    # Step 3: training config
    if "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "l2":
        intensity_loss = MSELoss()
        if rank == 0:
            print("Use l2 loss")
    else:
        intensity_loss = L1Loss()
        if rank == 0:
            print("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)

    adv_weight = 0.01
    perceptual_weight = args.autoencoder_train["perceptual_weight"]
    # kl_weight: important hyper-parameter.
    #     If too large, decoder cannot recon good results from latent space.
    #     If too small, latent space will not be regularized enough for the diffusion model
    kl_weight = args.autoencoder_train["kl_weight"]

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.autoencoder_train["lr"] * world_size)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.autoencoder_train["lr"] * world_size)

    # Step 4: training
    autoencoder_warm_up_n_epochs = 5
    n_epochs = args.autoencoder_train["n_epochs"]
    val_interval = args.autoencoder_train["val_interval"]
    best_val_recon_epoch_loss = 100.0
    total_step = 0

    for epoch in range(n_epochs):
        # train
        autoencoder.train()
        discriminator.train()
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)

            # train Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)

            recons_loss = intensity_loss(reconstruction, images)
            kl_loss = KL_loss(z_mu, z_sigma)
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # train Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()


        # validation
        if epoch % val_interval == 0:
            autoencoder.eval()
            val_recon_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)  # choose only one of Brats channels
                with torch.no_grad():
                    reconstruction, z_mu, z_sigma = autoencoder(images)
                    recons_loss = intensity_loss(
                        reconstruction.float(), images.float()
                    ) + perceptual_weight * loss_perceptual(reconstruction.float(), images.float())

                val_recon_epoch_loss += recons_loss.item()

            val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
            if rank == 0:
                # save last model
                print(f"Epoch {epoch} val_recon_loss: {val_recon_epoch_loss}")

                torch.save(autoencoder.state_dict(), trained_g_path_last)
                torch.save(discriminator.state_dict(), trained_d_path_last)
                # save best model
                if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                    best_val_recon_epoch_loss = val_recon_epoch_loss

                    torch.save(autoencoder.state_dict(), trained_g_path)
                    torch.save(discriminator.state_dict(), trained_d_path)
                    print("Got best val recon loss.")
                    print("Save trained autoencoder to", trained_g_path)
                    print("Save trained discriminator to", trained_d_path)



if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()