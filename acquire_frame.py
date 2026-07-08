import numpy as np
import os
from datetime import datetime
 
PIXEL_SIZE_UM = 17.0
PIXEL_SIZE_MM = PIXEL_SIZE_UM / 1000.0
 
def acquire_beam_frame(gd_ctrl, prefix="dataray", save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()
 
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = os.path.join(save_dir, f"{prefix}_{ts}")
 
    w = gd_ctrl.GetHorizontalPixels()
    h = gd_ctrl.GetVerticalPixels()
    data = gd_ctrl.GetWinCamDataAsVariant()
    frame = np.array(data, dtype=np.float64).reshape(h, w)
 
    frame_path = base + "_frame_raw.npy"
    bgsub_path = base + "_frame_bgsub.npy"
 
    np.save(frame_path, frame)
 
    border = np.concatenate([
        frame[0, :], frame[-1, :], frame[:, 0], frame[:, -1]
    ])
    bg = np.median(border)
 
    img = frame - bg
    img[img < 0] = 0
    np.save(bgsub_path, img)
 
    return {
        "base": base,
        "w": w,
        "h": h,
        "bg": bg,
        "frame_path": frame_path,
        "bgsub_path": bgsub_path,
    } import os
from datetime 