import torch
import pdb
from second.pytorch.core import box_torch_ops
from second.core import box_np_ops


def meshgrid(x, y, row_major=True):
    '''Return meshgrid in  range x& y

    Args:
        x: (int) first dim range
        y: (int) second dim range
        row_major: (bool) row major or column major

    Returns:
        meshgrid: (tensor) size[x*y,2]

    Example:
    >>meshgrid(3,2)     >>meshgrid(3,2,row_major=False)
    0 0                 0 0
    1 0                 0 1
    2 0                 0 2
    0 1                 1 0
    1 1                 1 1
    2 1                 1 2
    '''

    w = torch.arange(0, x)
    h = torch.arange(0, y)
    xx = w.repeat(y).view(-1, 1)
    yy = h.view(-1, 1).repeat(1, x).view(-1, 1)
    if row_major:
        xy = torch.cat([xx, yy], 1)
    else:
        xy = torch.cat([yy, xx], 1)
    return xy

def get_projected_idx(input_size, calib, img_shape, z_sel, rot_noise, scal_noise,grid_size=4.):
    '''Compute anchor boxes for each feature map.

    Args:
        input_size: (tensor) model input size of (w,h).

    Returns:
        boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                    where #anchors = fmw * fmh * #anchors_per_cell
    '''
    ## for FPN50 ##
    #        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(self.num_fms)]
    #        grid_size = [8., 16., 32., 64., 128.]
    ## for PIXOR ##
    fm_size = input_size

    fm_w, fm_h = int(fm_size[0]/grid_size), int(fm_size[1]/grid_size)
    xy2 = meshgrid(fm_w, fm_h).to(torch.float64) + 0.5
    xy = (xy2*grid_size).view(fm_w,fm_h,1,2).expand(fm_w,fm_h,1,2)

    xy = xy.to(torch.float32)
    z = torch.Tensor([z_sel]).view(1,1,1,1).expand(fm_w,fm_h,1,1)
    z = z.to(torch.float32)

    box = torch.cat([xy,z],3)
    anchor_boxes = box.view(-1,3)
    # Calculate Anchor Center
    anchor_center = torch.zeros(anchor_boxes.shape[0], 3, dtype=torch.float64)
    # anchor_center[:, 0] = 70.4 - (anchor_boxes[:, 0] / 10) ## x
    anchor_center[:, 0] = anchor_boxes[:, 0] / 10
    anchor_center[:, 1] = (anchor_boxes[:, 1] / 10) - 40. ##y
    anchor_center[:, 2] = anchor_boxes[:, 2] / 10

    # Convert to velodyne coordinates
    # anchor_center[:, 1] = -1 * anchor_center[:, 0]

    # Adjust center_z to center from bottom
    anchor_center[:, 2] += (1.52) / 2

    # Apply inverse augmentation
    # import pdb; pdb.set_trace()
    anchor_center_np = anchor_center.numpy()
    anchor_center_np = box_np_ops.rotation_points_single_angle(anchor_center_np, -rot_noise, axis=2)
    anchor_center_np *= 1./scal_noise

    # anchor_center_np = box_np_ops.rotation_points_single_angle(anchor_center_np, 1/scal_noise, axis=2)

    # import pdb; pdb.set_trace()
    anchor_center = torch.tensor(anchor_center_np, dtype=torch.float64)


    # # Get GT height
    # mask = ((max_ious>0.5)[0::2, ...].nonzero()*2).squeeze()
    # anchor_center[mask, 2] = -1 * boxes_[max_ids[mask], 2].to(torch.float64)
    # anchor_center[mask, 2] += (boxes_[max_ids[mask], 5].to(torch.float64)) / 2
    # anchor_center = anchor_center[0::2, ...]

    # Project to image space
    # pts_2d, pts_2d_norm = anchor_projector.point_to_image(anchor_center, data_dir)
    r_rect = torch.tensor(calib['rect'], dtype=torch.float32, device=torch.device("cpu")).to(torch.float64)
    P2 = torch.tensor(calib['P2'], dtype=torch.float32, device=torch.device("cpu")).to(torch.float64)
    velo2cam = torch.tensor(calib['Trv2c'], dtype=torch.float32, device=torch.device("cpu")).to(torch.float64)

    # anchor_center = anchor_center[:,[1,0,2]]
    anchor_center2 = box_torch_ops.lidar_to_camera(anchor_center,r_rect,velo2cam)
    idxs = box_torch_ops.project_to_image(anchor_center2, P2)
    # image_h = img_shape[2] ## 
    # image_w = img_shape[1]
    # img_shape_torch = torch.tensor([2496, 768]).to(torch.float64).view(1,2)
    img_shape_torch = torch.tensor([1248, 384]).to(torch.float64).view(1,2)
    idxs_norm = idxs / img_shape_torch
    # import pdb; pdb.set_trace()
    # idx = idxs_norm
    # # Filtering idx
    # mask = torch.mul(idx > 0, idx < 1).sum(dim=1) == 2
    # mask = mask.view(-1,1)

    # import pdb; pdb.set_trace()
    return idxs, idxs_norm