import torch
import os
import pytorch3d
# Notes
# Num views: 20, 30, 40 [10 failed]
# Resolution: 1/2 [1/4 failed]
from pytorch3d.io import load_obj, save_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import numpy as np
from pathlib import Path 

import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda:0")
SAMPLING_PTS = 5000
DATASET = 'DTU'
OBJ_IDS = ['110_rabbit','105_plush', '83_smurf', '114_buddha', '122_owl', '65_skull', '55_bunny', '106_birds', '40_block', '118_angel', '24_redhouse', '37_scissors', '69_snowman', '97_cans', '63_fruits'] # 63_fruits
# DATASET = 'OFR'
# OBJ_IDS = ['ofr_4','ofr_32','ofr_79','ofr_64','ofr_100']
# EXPERIMENT_SUFFIXES = ['']
EXPERIMENT_SUFFIXES = ['', '_scale_half']
# EXPERIMENT_SUFFIXES = ['', '_random_001', '_random_0001', '_random_00001']
label_map = {}
for exp_name in EXPERIMENT_SUFFIXES:
    if len(exp_name) == 0:
        label_map[exp_name] = "Full Resolution"
    else:
        keys = exp_name.split("_")
        label_map[exp_name] = "{} {}".format(keys[1], keys[2])

def load_mesh(path):
    if path.suffix == '.obj':
        verts, faces, aux = load_obj(str(path))
        faces_idx = faces.verts_idx.to(device)
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        verts = verts.to(device)
        trg_mesh = Meshes(verts=[verts], faces=[faces_idx])
    elif path.suffix == '.ply':
        assert path.exists(), "Path {} does not exist".format(path)
        verts, faces = load_ply(str(path))
        center = verts.mean(0)
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        verts.to(device)
        faces.to(device)
        trg_mesh = Meshes(verts=[verts], faces=[faces])
    else:
        print("Unsupported suffix:", path.suffix)
    return trg_mesh

def get_chamfer_loss(gt_mesh, pred_mesh, gt_is_ply=False):
    # We sample 5k points from the surface of each mesh 
    if gt_is_ply:
        gt_vertices = gt_mesh.verts_list()[0] #sample_points_from_meshes(gt_mesh, SAMPLING_PTS)
        random_verts = torch.randperm(SAMPLING_PTS)
        sample_trg = gt_vertices[random_verts,:].reshape((1,SAMPLING_PTS,3)).to(device)
    else:
        sample_trg = sample_points_from_meshes(gt_mesh, SAMPLING_PTS)
    sample_src = sample_points_from_meshes(pred_mesh, SAMPLING_PTS)

    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    return loss_chamfer

iterations = None
exp_data_cache = {}
for OBJ_ID in tqdm(OBJ_IDS):
    GT_OBJ_ID = "_".join(OBJ_ID.split("_")[:2])
    if DATASET == 'DTU':
        GT_MESH_PATH = Path("/viscam/u/tanmayx/projects/data") / GT_OBJ_ID / 'gt.ply'
    else:
        GT_MESH_PATH = Path("/viscam/u/tanmayx/projects/data") / GT_OBJ_ID / 'model.obj'
    gt_mesh = load_mesh(GT_MESH_PATH)
    
    for exp_suffix in EXPERIMENT_SUFFIXES:
        exp_id = OBJ_ID + exp_suffix
        if not exp_suffix in exp_data_cache:
            exp_data_cache[exp_suffix] = np.zeros((21,), dtype=float)
        all_losses = []
        src_mesh_dir = Path("/viscam/u/tanmayx/projects/neural-deferred-shading/out") / str(exp_id) / "meshes"
        if not src_mesh_dir.exists():
            continue
        mesh_paths = sorted([p for p in src_mesh_dir.iterdir() if p.suffix == '.obj'])
        if iterations is None:
            iterations = [int(mesh_p.stem.split('_')[-1]) for mesh_p in mesh_paths]
        for mesh_p in mesh_paths: 
            pred_mesh = load_mesh(mesh_p)
            loss_chamfer = get_chamfer_loss(gt_mesh, pred_mesh, GT_MESH_PATH.suffix == '.ply')
            all_losses.append(loss_chamfer.cpu().numpy())
        # if len(all_losses) == 0:
        #     breakpoint()     
        exp_data_cache[exp_suffix] += np.array(all_losses)
# Normalize each exp by number of objects
for exp_suffix in EXPERIMENT_SUFFIXES:
    exp_data_cache[exp_suffix] /= len(OBJ_IDS)
    # plt.plot(iters, loss, label=label_map[OBJ_ID])
    
# Plot experiment vs iteration
for exp_suffix in EXPERIMENT_SUFFIXES:
    plt.plot(iterations, exp_data_cache[exp_suffix], label=label_map[exp_suffix])

plt.ylabel("Chamfer Distance")
plt.xlabel("Iteration")
plt.legend()
plt.savefig("chamfer_distances_low_res.png")
