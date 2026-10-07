# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from time import time
from models import SiT_models
from dataset import CustomDataset, MemmapLatentDataset
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from train_utils import parse_transport_args
import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}-{int(time())}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        if args.wandb:
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        query_seqlen=args.query_len,
        aux_recon=args.aux_recon,   # A = PE-only (default off); B = --aux-recon (bottleneck-AE loss)
        compressor_version=args.compressor_version,   # 1 = fixed; 2 = soft adaptive; 3 = hard top-k
        keep_ratio=args.keep_ratio,                   # V3 only
        learn_sigma=False
    )
    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        args = state_dict["args"]
    model = torch.compile(model.to(device))
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.requires_grad_(False)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # Use the fast uncompressed memmap if it exists (see repack_latents.py); else
    # fall back to the per-file .npz tree.
    if os.path.exists(os.path.join(args.data_path, "latents.npy")):
        dataset = MemmapLatentDataset(args.data_path)
        logger.info("Using MemmapLatentDataset (uncompressed memmap).")
    else:
        dataset = CustomDataset(args.data_path)
        logger.info("Using CustomDataset (.npz tree) — consider repack_latents.py for speed.")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
        persistent_workers=args.num_workers > 0,   # don't respawn workers every epoch
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module._orig_mod if hasattr(model.module, '_orig_mod') else model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_aux = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    n = 32
    ys = torch.randint(1000, size=(n,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)
    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            opt.zero_grad()
            
            x = x.to(device)
            y = y.to(device)
            x = DiagonalGaussianDistribution(x).sample().mul_(0.18215)
            # with torch.inference_mode():
            #     xis = []
            #     # Map input images to latent space + normalize latents:
            #     for i in range(0, x.shape[0], 32):
            #         xi = vae.encode(x[i:i+32]).latent_dist.sample().mul_(0.18215)
            #         xis.append(xi)
            #     x = torch.cat(xis)
            model_kwargs = dict(y=y)
            # bf16 autocast: model runs in bf16 (big throughput win on Blackwell),
            # master weights stay fp32 in AdamW. No GradScaler needed for bf16.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"]
            aux_recon = loss_dict.get("aux_recon_loss")

            loss.backward()
            # measure total grad norm (max_norm huge => pure measurement, no actual clipping)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9).item()

            opt.step()
            update_ema(ema, model.module._orig_mod if hasattr(model.module, '_orig_mod') else model.module)

            # Log loss values:
            running_loss += loss.item()
            running_aux += aux_recon.item() if aux_recon is not None else 0.
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_aux = torch.tensor(running_aux / log_steps, device=device)
                dist.all_reduce(avg_aux, op=dist.ReduceOp.SUM)
                avg_aux = avg_aux.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Aux Recon: {avg_aux:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "aux recon loss": avg_aux, "train steps/sec": steps_per_sec, "norm": norm },
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                running_aux = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps % args.sample_every == 0 and train_steps > 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_fn = transport_sampler.sample_ode() # default to ode sampling
                    samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

                    if use_cfg: #remove null samples
                        samples, _ = samples.chunk(2, dim=0)

                    # [vram] decode in sub-batches: the 256px SD-VAE decoder is the memory
                    # hotspot, so a single decode of all samples is a large one-off spike.
                    samples = samples / 0.18215
                    _dec_chunk = 8
                    decoded = [vae.decode(samples[i:i + _dec_chunk]).sample.float()
                               for i in range(0, samples.shape[0], _dec_chunk)]
                    samples = torch.cat(decoded, dim=0)
                    del decoded
                    out_samples = torch.zeros((32 * dist.get_world_size(), 3, args.image_size, args.image_size), device=device)
                    dist.all_gather_into_tensor(out_samples, samples)

                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                # [vram] release the sampling transients so the caching allocator doesn't hold
                # the peak reserved for the rest of training (the 8.5->15GB plateau).
                del samples, out_samples
                torch.cuda.empty_cache()
                logging.info("Generating EMA samples done.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5_000)
    parser.add_argument("--sample-every", type=int, default=100_100)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--query-len", type=int, default=32, required=False)
    parser.add_argument("--aux-recon", action="store_true",
                        help="Enable bottleneck-AE reconstruction loss (run B). Off = run A, PE-only.")
    parser.add_argument("--compressor-version", type=int, default=1, choices=[1, 2, 3],
                        help="1 = fixed pool+xattn (default); 2 = soft content-adaptive; 3 = hard top-k routing.")
    parser.add_argument("--keep-ratio", type=float, default=0.5,
                        help="V3 only: fraction of input tokens kept by the top-k router.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
