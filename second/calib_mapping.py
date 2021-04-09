import os
from glob import glob
import pdb 
import numpy as np
from tqdm import tqdm
# P0: cam_to_cam, line10, P_rect_00
# P1: cam_to_cam, line18, P_rect_01
# P2: cam_to_cam, line26, P_rect_02
# P3: cam_to_cam, line34, P_rect_03
# R0_rect: cam_to_cam, line9, R_rect_00
# Tr_velo_to_cam: velo_to_cam line2 [3,3], line3[3,1] [R,T]
# Tr_imu_to_velo: imu_to_velo line2 [3,3], line3[3,1] [R,T]

scene_num = '0095'
root_dir = '/mnt/sdb/jhyoo/dataset/KITTI_raw'
targ_dir = os.path.join(root_dir,'2011_09_26_drive_'+scene_num+'_sync','testing/calib/')
calib_dir = os.path.join(root_dir,'2011_09_26_drive_'+scene_num+'_sync','2011_09_26')
idx_ref = os.path.join(root_dir,'2011_09_26_drive_'+scene_num+'_sync','testing/image_2/')
idx_list= glob(os.path.join(idx_ref,'*.png'))
imgset_dir = os.path.join(root_dir,'2011_09_26_drive_'+scene_num+'_sync','testing')


cam_to_cam = open(os.path.join(calib_dir,'calib_cam_to_cam.txt'), 'r')
velo_to_cam = open(os.path.join(calib_dir,'calib_velo_to_cam.txt'), 'r')
imu_to_velo = open(os.path.join(calib_dir,'calib_imu_to_velo.txt'), 'r')

cam_to_cam = cam_to_cam.read().splitlines() 
velo_to_cam = velo_to_cam.read().splitlines() 
imu_to_velo = imu_to_velo.read().splitlines() 
P0 = cam_to_cam[9].split(" ")[1:13]
P1 = cam_to_cam[17].split(" ")[1:13]
P2 = cam_to_cam[25].split(" ")[1:13]
P3 = cam_to_cam[33].split(" ")[1:13]
R0_rect = cam_to_cam[8].split(" ")[1:10]

R = np.array(velo_to_cam[1].split(" ")[1:10]).reshape(3,3)
T = np.array(velo_to_cam[2].split(" ")[1:4]).reshape(3,1)
Tr_velo_to_cam = np.concatenate((R,T),1).flatten()

R = np.array(imu_to_velo[1].split(" ")[1:10]).reshape(3,3)
T = np.array(imu_to_velo[2].split(" ")[1:4]).reshape(3,1)
Tr_imu_to_velo = np.concatenate((R,T),1).flatten()
data = 'P0: '+ ' '.join(P0)+'\n'
data += 'P1: '+ ' '.join(P1)+'\n'
data += 'P2: '+ ' '.join(P2)+'\n'
data += 'P3: '+ ' '.join(P3)+'\n'
data += 'R0_rect: '+ ' '.join(R0_rect)+'\n'
data += 'Tr_velo_to_cam: '+ ' '.join(Tr_velo_to_cam)+'\n'
data += 'Tr_imu_to_velo: '+ ' '.join(Tr_imu_to_velo)+'\n'


f_names = ''
for i in tqdm(idx_list):
    f_name= i.split('/')[-1][:10]
    f_names += f_name+'\n'
    calib_file = open(os.path.join(targ_dir,f_name+'.txt'), 'w')
    calib_file.write(data)
    calib_file.close()

imgset_file = open(os.path.join(imgset_dir,'test.txt'), 'w')
imgset_file.write(f_names)
imgset_file.close()
imgset_dir
temp = 1