import torch
import math
import os
import sys
import torch.nn as nn
import torch.nn.functional as F



proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from utils.ChamferDistancePytorch.fscore import fscore


def calc_cd(output, gt, calc_f1=False):
    cham_loss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, _, _ = cham_loss(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t
