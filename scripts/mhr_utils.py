"""MHR forward helpers in the Track 1 frame (official MHR v1.0.1 TorchScript model).

Track 1 / scorer convention:
    verts_m = diag(1, -1, -1) @ MHR(shape, [pose | scales], expr=0, correctives=on) / 100
pose[:, 0:3] is the root translation in MHR parameter units (1 unit = 10 cm), expressed in
MHR's native frame (y and z flipped w.r.t. the OpenCV camera frame).
"""
import numpy as np
import torch

DEFAULT_MHR = "/work/doreen071/v2d/assets/mhr_model.pt"
FLIP = np.array([1.0, -1.0, -1.0])
TRANS_UNIT_M = 0.1  # one root-translation parameter unit, in metres


def cam_t_to_root_trans(cam_t):
    """Camera-frame offset (metres, OpenCV axes) -> pose[:, 0:3] increment."""
    return np.asarray(cam_t) * FLIP / TRANS_UNIT_M


class MHRForward:
    def __init__(self, path=DEFAULT_MHR, device="cuda", dtype=torch.float32):
        self.device, self.dtype = torch.device(device), dtype
        self.model = torch.jit.load(path, map_location=self.device).to(dtype).eval()
        self.flip = torch.tensor(FLIP, dtype=dtype, device=self.device)

    @torch.no_grad()
    def __call__(self, pose, scales, shape, chunk=256):
        """pose [T,136], scales [T,68] or [68], shape [T,45] or [45] -> (verts [T,V,3], joints [T,127,3]) metres, Track 1 frame."""
        pose = torch.as_tensor(np.asarray(pose), dtype=self.dtype, device=self.device).reshape(-1, 136)
        T = len(pose)
        scales = torch.as_tensor(np.asarray(scales), dtype=self.dtype, device=self.device).reshape(-1, 68).expand(T, -1)
        shape = torch.as_tensor(np.asarray(shape), dtype=self.dtype, device=self.device).reshape(-1, 45).expand(T, -1)
        vs, js = [], []
        for a in range(0, T, chunk):
            p = torch.cat([pose[a:a + chunk], scales[a:a + chunk]], 1)
            expr = torch.zeros(len(p), 72, dtype=self.dtype, device=self.device)
            v, sk = self.model(shape[a:a + chunk].contiguous(), p, expr, True)
            vs.append(v * self.flip / 100.0)
            js.append(sk[..., :3] * self.flip / 100.0)
        return torch.cat(vs).cpu().numpy(), torch.cat(js).cpu().numpy()


def project(points, fx, fy, cx, cy):
    z = points[..., 2]
    return np.stack([fx * points[..., 0] / z + cx, fy * points[..., 1] / z + cy], -1)
