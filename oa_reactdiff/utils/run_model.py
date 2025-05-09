import torch
import time
import numpy as np
import ase
from pathlib import Path
from ase.io import read, write
from oa_reactdiff.trainer.pl_trainer import DDPMModule
from oa_reactdiff.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule
from oa_reactdiff.diffusion._normalizer import FEATURE_MAPPING
from oa_reactdiff.utils.sampling_tools import write_tmp_xyz

def onehot_convert(atomic_numbers):
    """
    Convert a list of atomic numbers into an one-hot matrix
    """
    encoder= {1: [1, 0, 0, 0, 0], 6: [0, 1, 0, 0, 0], 7: [0, 0, 1, 0, 0], 8: [0, 0, 0, 1, 0]}
    onehot = [encoder[i] for i in atomic_numbers]
    return np.array(onehot)

def parse_xyz(xyz, device):
    """
    Function used to parse a xyz file and convert it into input entry format
    """
    representation = {}
    mol = read(xyz)
    atomic_numbers = mol.get_atomic_numbers()
    coordinates = mol.get_positions()
    representation['size']   = torch.tensor(np.array([len(atomic_numbers)]), dtype=torch.int64, device=device)
    representation['pos']    = torch.tensor(coordinates, dtype=torch.float32, device=device)
    representation['one_hot']= torch.tensor(onehot_convert(atomic_numbers), dtype=torch.int64, device=device)
    representation['charge'] = torch.tensor(np.array([[i] for i in atomic_numbers]), dtype=torch.int32, device=device)
    representation['mask'] = torch.tensor(np.zeros(len(atomic_numbers)), dtype=torch.int64, device=device)
    return representation

def parse_rxn_input(rxyz, pxyz, device):
    """
    Function used to parse reactant and product xyz files and convert into input entry
    """
    r_rep =  parse_xyz(rxyz, device)
    p_rep =  parse_xyz(pxyz, device)
    return [r_rep, r_rep, p_rep]

def pred_ts(rxyz,pxyz,output_path,timesteps=150):
    """
    Apply Oa-Reactdiff to provide a TS structure based on input R and P
    """
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # define device
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    # load model
    ddpm_trainer = DDPMModule.load_from_checkpoint(
        checkpoint_path="pretrained-ts1x-diff.ckpt",
        map_location=device,
    )

    noise_schedule: str = "polynomial_2"
    precision: float = 1e-5
    gamma_module = PredefinedNoiseSchedule(
        noise_schedule=noise_schedule,
        timesteps=timesteps,
        precision=precision,
    )
    schedule = DiffSchedule(
        gamma_module=gamma_module,
        norm_values=ddpm_trainer.ddpm.norm_values
    )
    ddpm_trainer.ddpm.schedule = schedule
    ddpm_trainer.ddpm.T = timesteps
    ddpm_trainer = ddpm_trainer.to(device)

    # parse input rxn
    representations = parse_rxn_input(rxyz, pxyz, device=device)
    n_samples = representations[0]["size"].size(0)
    fragments_nodes = [repre["size"] for repre in representations]
    conditions = torch.tensor([[0] for _ in range(n_samples)], device=device)

    xh_fixed = [
        torch.cat(
            [repre[feature_type] for feature_type in FEATURE_MAPPING],
            dim=1,
        )
        for repre in representations
    ]

    # run inpaint
    out_samples, out_masks = ddpm_trainer.ddpm.inpaint(
        n_samples=n_samples,
        fragments_nodes=fragments_nodes,
        conditions=conditions,
        return_frames=1,
        resamplings=5,
        jump_length=5,
        timesteps=None,
        xh_fixed=xh_fixed,
        frag_fixed=[0, 2],  # r and p
    )

    # write down xyz
    write_tmp_xyz(
        fragments_nodes,
        out_samples[0],
        idx=[1],
        localpath=output_path
    )
    return