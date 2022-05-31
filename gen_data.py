import os
import cv2
import numpy as np
from tqdm import tqdm
from pyntcloud import PyntCloud

INTENSITY_2_GRAY_RATIO = 255. / 65535.


def load_road_surface(seq_dir, sample_rate = None):
    pcd = PyntCloud.from_file(os.path.join(seq_dir, 'roadsurface.ply'))
    x = np.asarray(pcd.points['x'])
    y = np.asarray(pcd.points['y'])
    z = np.asarray(pcd.points['z'])
    intensity = np.asarray(pcd.points['scalar_Intensity'])
    intensity = (intensity * INTENSITY_2_GRAY_RATIO).astype(np.uint8)
    xyz = np.stack((x, y, z), axis = 1)

    if sample_rate is not None:
        sample_num = int(len(xyz) * sample_rate)
        idxs = np.random.permutation(len(xyz))[:sample_num]
        xyz = xyz[idxs]
        intensity = intensity[idxs]

    return xyz, intensity

def load_road_marking(seq_dir):
    pcd = PyntCloud.from_file(os.path.join(seq_dir, 'roadmarking.ply'))
    x = np.asarray(pcd.points['x'])
    y = np.asarray(pcd.points['y'])
    z = np.asarray(pcd.points['z'])
    xyz = np.stack((x, y, z), axis = 1)
    return xyz

def surface_to_image(points, intensity, grid_size = 0.04):
    min_point = np.min(points, axis = 0)
    max_point = np.max(points, axis = 0)

    img_W = int(np.ceil((max_point[0] - min_point[0]) / grid_size))
    img_H = int(np.ceil((max_point[1] - min_point[1]) / grid_size))

    image = np.zeros((img_H, img_W), np.uint8)
    mask = np.zeros((img_H, img_W), dtype = np.bool_)
    for i in tqdm(range(len(points))):
        x_idx = int(np.floor((points[i][0] - min_point[0]) / grid_size))
        x_idx = min(x_idx, img_W - 1)
        y_idx = int(np.floor((points[i][1] - min_point[1]) / grid_size))
        y_idx = min(y_idx, img_H - 1)

        image[y_idx, x_idx] = max(image[y_idx, x_idx], intensity[i])
        mask[y_idx][x_idx] = True

    return image, mask, min_point

def marking_to_image(points, mask, min_point, grid_size):
    img_H, img_W = mask.shape[0], mask.shape[1]

    U = np.floor((points[:, 1] - min_point[1]) / grid_size).astype(np.int32)
    U = np.clip(U, 0, img_H - 1)
    V = np.floor((points[:, 0] - min_point[0]) / grid_size).astype(np.int32)
    V = np.clip(V, 0, img_W - 1)

    image = np.zeros((img_H, img_W), dtype = np.uint8)
    for u_, v_ in zip(U, V):
        if mask[u_, v_]:
            image[u_, v_] = 255
    return image

def equalizeHist(image, mask):
    image_gray = image[mask]
    hist = np.zeros((256), dtype = np.uint32)
    for gray in image_gray:
        hist[gray] += 1
    
    cumsum_hist = {}
    for gray in range(256):
        if hist[gray] == 0:
            continue
        cumsum_hist[gray] = np.sum(hist[:gray]) + hist[gray]
    temp = [cumsum_hist[_] for _ in cumsum_hist.keys()]
    min_cnt = np.min(temp).item()
    max_cnt = np.max(temp).item()

    for u in range(mask.shape[0]):
        for v in range(mask.shape[1]):
            if mask[u, v]:
                value = image[u, v]
                value = int((cumsum_hist[value] - min_cnt) / (max_cnt - min_cnt) * 255)
                image[u, v] = value
    
    return image.astype(np.uint8)

def gen_patch(surface, marking, mask, patch_size, shift_step, surface_dir, marking_dir, threshold = 0.005):
    img_H, img_W = mask.shape[0], mask.shape[1]
    patch_pixel_num = patch_size * patch_size

    for u in range(0, img_H, shift_step):
        for v in range(0, img_W, shift_step):
            bgn_u, bgn_v = u, v
            end_u, end_v = u + patch_size, v + patch_size
            if end_u > img_H or end_v > img_W:
                end_u, end_v = img_H, img_W
                bgn_u, bgn_v = end_u - patch_size, end_v - patch_size
            

            patch_mask = mask[bgn_u:end_u, bgn_v:end_v]
            if patch_mask.astype(np.uint8).sum() / patch_pixel_num < threshold:
                continue

            patch_surface = surface[bgn_u:end_u, bgn_v:end_v]
            # patch_surface = equalizeHist(patch_surface, patch_mask)
            patch_marking = marking[bgn_u:end_u, bgn_v:end_v]

            filename = f'{bgn_u}_{bgn_v}.png'
            cv2.imwrite(os.path.join(surface_dir, filename), patch_surface)
            cv2.imwrite(os.path.join(marking_dir, filename), patch_marking)


def process(root_dir, seq, save_dir, grid_size = 0.04, patch_size = 256, shift_step = 256):

    seq_dir = os.path.join(root_dir, f'{seq}')
    save_seq_dir = os.path.join(save_dir, f'{seq}')
    surface_dir = os.path.join(save_seq_dir, 'surface')
    marking_dir = os.path.join(save_seq_dir, 'marking')
    os.makedirs(surface_dir)
    os.makedirs(marking_dir)

    print('加载路面点云')
    surface, intensity = load_road_surface(seq_dir)
    print('记载道路标识')
    marking = load_road_marking(seq_dir)
    print('路面点云投影')
    surface_image, surface_mask, min_point = surface_to_image(surface, intensity, grid_size)
    print('道路标识投影')
    marking_image = marking_to_image(marking, surface_mask, min_point, grid_size)

    cv2.imwrite(os.path.join(save_seq_dir, 'surface.png'), surface_image)
    cv2.imwrite(os.path.join(save_seq_dir, 'marking.png'), marking_image)

    print('分块')
    gen_patch(surface_image, marking_image, surface_mask, patch_size, shift_step, surface_dir, marking_dir)

if __name__ == '__main__':
    root_dir = '../test数据生成(road2、11)/'
    grid_size = 0.04
    patch_size = 256
    shift_step = 256
    save_dir = './test6and10'

    for seq in range(2):
        process(root_dir, seq, save_dir, grid_size, patch_size, shift_step)


