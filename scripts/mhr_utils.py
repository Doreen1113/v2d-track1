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

    @torch.no_grad()
    def skeleton(self, pose, scales, shape, joints, vert_ids=None, chunk=256):
        """Global joint frames in the Track 1 frame.

        Returns positions [T,J,3] (metres), rotations [T,J,3,3] (columns = joint axes in the
        Track 1 frame, i.e. diag(1,-1,-1) @ R_mhr) and, if vert_ids is given, verts [T,len(ids),3].
        The skeleton state is (position cm, quaternion xyzw, scale) per joint.
        """
        from scipy.spatial.transform import Rotation
        pose = torch.as_tensor(np.asarray(pose), dtype=self.dtype, device=self.device).reshape(-1, 136)
        T = len(pose)
        scales = torch.as_tensor(np.asarray(scales), dtype=self.dtype, device=self.device).reshape(-1, 68).expand(T, -1)
        shape = torch.as_tensor(np.asarray(shape), dtype=self.dtype, device=self.device).reshape(-1, 45).expand(T, -1)
        joints = list(joints)
        ps, qs, vs = [], [], []
        for a in range(0, T, chunk):
            p = torch.cat([pose[a:a + chunk], scales[a:a + chunk]], 1)
            expr = torch.zeros(len(p), 72, dtype=self.dtype, device=self.device)
            v, sk = self.model(shape[a:a + chunk].contiguous(), p, expr, True)
            sk = sk[:, joints].double().cpu().numpy()
            ps.append(sk[..., :3] * FLIP / 100.0)
            qs.append(sk[..., 3:7])
            if vert_ids is not None:
                vs.append((v[:, vert_ids] * self.flip / 100.0).cpu().numpy())
        pos = np.concatenate(ps)
        q = np.concatenate(qs).reshape(-1, 4)
        rot = (np.diag(FLIP) @ Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix())
        rot = rot.reshape(T, len(joints), 3, 3)
        return pos, rot, (np.concatenate(vs) if vert_ids is not None else None)


def project(points, fx, fy, cx, cy):
    z = points[..., 2]
    return np.stack([fx * points[..., 0] / z + cx, fy * points[..., 1] / z + cy], -1)
