import torch
import numpy as np
import cv2
import open3d as o3d
import yaml
import os
from tqdm import tqdm

def load_calibration_from_yaml(yaml_path, camera_name):
    """从YAML加载指定相机的内参和外参（相机到雷达的变换矩阵）"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    cam_params = config[camera_name]
    K_rect = np.array(cam_params['K_rect'], dtype=np.float32)
    T_cam_to_lidar = np.array(cam_params['T_cam_lidar'], dtype=np.float32)  # 相机到雷达的变换矩阵
    img_width, img_height = cam_params['width'], cam_params['height']
    return K_rect, T_cam_to_lidar, img_width, img_height

def match_timestamps(pointcloud_dir, image_dir, max_time_diff=1.0, debug=False):
    """按时间戳匹配点云与图像文件，正确提取带小数的时间戳"""
    # 正确提取时间戳：假设文件名格式为"时间戳.后缀"，例如"12345.678.jpg"
    def extract_timestamp(filename):
        # 分割文件名与扩展名（处理多级后缀，如.tar.gz）
        name_without_ext = os.path.splitext(filename)[0]
        # 时间戳可能包含小数点，因此直接使用完整的文件名作为时间戳（去除扩展名后）
        return name_without_ext
    
    # 加载所有文件并提取时间戳（保留完整字符串，包括小数）
    pcd_files = {extract_timestamp(f): f for f in os.listdir(pointcloud_dir) if f.endswith('.pcd')}
    cam0_imgs = {extract_timestamp(f): f for f in os.listdir(os.path.join(image_dir, 'cam0')) if f.endswith('.jpg')}
    cam1_imgs = {extract_timestamp(f): f for f in os.listdir(os.path.join(image_dir, 'cam1')) if f.endswith('.jpg')}
    
    if debug:
        print(f"点云时间戳示例: {next(iter(pcd_files.keys())) if pcd_files else None}")
        print(f"相机0时间戳示例: {next(iter(cam0_imgs.keys())) if cam0_imgs else None}")
    
    matches = []
    for pcd_ts_str in pcd_files:
        pcd_ts_float = float(pcd_ts_str)
        cam_ts_str = None
        
        # 1. 精确匹配（时间戳字符串完全一致）
        if pcd_ts_str in cam0_imgs and pcd_ts_str in cam1_imgs:
            cam_ts_str = pcd_ts_str
            if debug:
                print(f"精确匹配: {pcd_ts_str}")
        
        # 2. 最近邻匹配（允许时间差≤max_time_diff）
        else:
            common_timestamps = set(cam0_imgs.keys()) & set(cam1_imgs.keys())
            if common_timestamps:
                # 转换为浮点数进行比较，保留原始字符串
                nearest_ts_str = min(common_timestamps, 
                                    key=lambda x: abs(float(x) - pcd_ts_float))
                diff = abs(float(nearest_ts_str) - pcd_ts_float)
                if diff <= max_time_diff:
                    cam_ts_str = nearest_ts_str
                    if debug:
                        print(f"最近邻匹配: {pcd_ts_str} → {nearest_ts_str} (diff={diff:.6f}s)")
        
        # 3. 仅当匹配有效时添加结果
        if cam_ts_str:
            matches.append({
                'pcd_path': os.path.join(pointcloud_dir, pcd_files[pcd_ts_str]),
                'cam0_img': os.path.join(image_dir, 'cam0', cam0_imgs[cam_ts_str]),
                'cam1_img': os.path.join(image_dir, 'cam1', cam1_imgs[cam_ts_str]),
                'pcd_timestamp': pcd_ts_str,
                'cam_timestamp': cam_ts_str
            })
    return matches


def transform_torch(points_3d, T):
    """坐标变换：应用变换矩阵 T（输入点云坐标系 → 输出点云坐标系）"""
    points_homo = torch.cat([points_3d, torch.ones((points_3d.shape[0], 1), device=points_3d.device)], dim=1)
    return torch.matmul(points_homo, T.T)[:, :3]

def project_points(points_lidar_torch, K, T_cam_to_lidar, img_size, min_depth, max_depth):
    """执行投影：雷达坐标系 → 相机坐标系 → 像素平面"""
    device = points_lidar_torch.device
    img_width, img_height = img_size
    
    # 计算雷达到相机的变换矩阵（T_lidar_to_cam = T_cam_to_lidar 的逆矩阵）
    T_lidar_to_cam = torch.inverse(torch.tensor(T_cam_to_lidar, dtype=torch.float32, device=device))
    
    # 1. 雷达坐标系 → 相机坐标系（得到非齐次坐标 [X, Y, Z]）
    points_cam = transform_torch(points_lidar_torch, T_lidar_to_cam)  # 形状: (N, 3)
    
    # 2. 透视投影：相机坐标系 → 像素平面（直接使用非齐次坐标计算）
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    u = (K[0, 0] * X + K[0, 2] * Z) / Z  # u = fx * (X/Z) + cx
    v = (K[1, 1] * Y + K[1, 2] * Z) / Z  # v = fy * (Y/Z) + cy
    depth = Z  # 相机坐标系下的深度值
    
    # 3. 有效性掩码
    valid_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    valid_mask &= (depth > min_depth) & (depth < max_depth)
    
    # 4. 构建深度图（每个像素取最近点深度）
    depth_map = torch.full((img_height, img_width), torch.inf, device=device)
    u_valid = u[valid_mask].long()
    v_valid = v[valid_mask].long()
    depth_valid = depth[valid_mask]
    
    # 使用 scatter_reduce 高效更新深度图（取最小值）
    indices_1d = v_valid * img_width + u_valid
    depth_map.view(-1).scatter_reduce_(0, indices_1d, depth_valid, reduce='min', include_self=False)
    
    return valid_mask, u_valid, v_valid, depth_map

def main():
    yaml_path = "config.yaml"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    pointcloud_dir = config['pointcloud_path']
    image_dir = config['image_path']
    output_path = config['output_path']
    min_depth = config['min_depth']
    max_depth = config['max_depth']
    
    # 启用调试模式以查看时间戳匹配情况
    debug_mode = True
    
    # 创建输出目录
    os.makedirs(os.path.join(output_path, 'colored-cloud-points'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'depth-images/cam0'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'depth-images/cam1'), exist_ok=True)
    
    # 加载相机参数
    cam0_K, cam0_T, cam0_w, cam0_h = load_calibration_from_yaml(yaml_path, 'cam0')
    cam1_K, cam1_T, cam1_w, cam1_h = load_calibration_from_yaml(yaml_path, 'cam1')
    
    # 转换为张量
    cam0_K_torch = torch.tensor(cam0_K, dtype=torch.float32).to('cuda')
    cam1_K_torch = torch.tensor(cam1_K, dtype=torch.float32).to('cuda')
    
    matches = match_timestamps(pointcloud_dir, image_dir, debug=debug_mode)
    print(f"找到 {len(matches)} 对匹配的点云与图像")
    
    for match in tqdm(matches, desc="Processing"):
        pcd_ts = match['pcd_timestamp']  # 点云时间戳（字符串格式）
        cam_ts = match['cam_timestamp']  # 图像时间戳（字符串格式）
        pcd_path = match['pcd_path']
        cam0_img_path = match['cam0_img']
        cam1_img_path = match['cam1_img']
        
        if debug_mode:
            print(f"\n处理数据对:")
            print(f"  点云时间戳: {pcd_ts}")
            print(f"  图像时间戳: {cam_ts}")
            print(f"  点云路径: {pcd_path}")
            print(f"  相机0图像路径: {cam0_img_path}")
            print(f"  相机1图像路径: {cam1_img_path}")
        
        # 加载雷达点云（雷达坐标系）
        pcd = o3d.io.read_point_cloud(pcd_path)
        points_lidar_np = np.asarray(pcd.points).astype(np.float32)
        points_lidar_torch = torch.tensor(points_lidar_np, dtype=torch.float32, device='cuda')
        
        # --------------------------- 处理 cam0 ---------------------------
        img0_cv = cv2.imread(cam0_img_path)
        img0_rgb = cv2.cvtColor(img0_cv, cv2.COLOR_BGR2RGB)
        img0_torch = torch.tensor(img0_rgb, dtype=torch.float32, device='cuda').permute(2, 0, 1) / 255.0  # (C, H, W)
        
        # 投影获取有效点和深度图
        valid_mask0, u0_valid, v0_valid, depth_map0 = project_points(
            points_lidar_torch, cam0_K_torch, cam0_T, (cam0_w, cam0_h), min_depth, max_depth
        )
        
        # 提取颜色（仅有效点）
        rgb0 = torch.zeros((points_lidar_torch.shape[0], 3), device='cuda')
        if valid_mask0.sum() > 0:
            rgb0[valid_mask0] = img0_torch[:, v0_valid, u0_valid].permute(1, 0)  # (N, C)
        
        # --------------------------- 处理 cam1 ---------------------------
        img1_cv = cv2.imread(cam1_img_path)
        img1_rgb = cv2.cvtColor(img1_cv, cv2.COLOR_BGR2RGB)
        img1_torch = torch.tensor(img1_rgb, dtype=torch.float32, device='cuda').permute(2, 0, 1) / 255.0  # (C, H, W)
        
        valid_mask1, u1_valid, v1_valid, depth_map1 = project_points(
            points_lidar_torch, cam1_K_torch, cam1_T, (cam1_w, cam1_h), min_depth, max_depth
        )
        
        rgb1 = torch.zeros((points_lidar_torch.shape[0], 3), device='cuda')
        if valid_mask1.sum() > 0:
            rgb1[valid_mask1] = img1_torch[:, v1_valid, u1_valid].permute(1, 0)  # (N, C)
        
        # --------------------------- 合并颜色 ---------------------------
        rgb0_cpu = rgb0.cpu().numpy()
        rgb1_cpu = rgb1.cpu().numpy()
        valid0 = (rgb0_cpu[:, 0] > 0).astype(bool)  # cam0有效颜色点
        valid1 = (rgb1_cpu[:, 0] > 0).astype(bool)  # cam1有效颜色点
        
        # 优先使用cam0颜色，cam0未覆盖的点使用cam1颜色
        colored_points = np.zeros_like(points_lidar_np)
        colored_points[valid0] = rgb0_cpu[valid0]
        colored_points[~valid0 & valid1] = rgb1_cpu[valid1 & ~valid0]
        
        # --------------------------- 保存结果 ---------------------------
        # 保存彩色点云（文件名与原始点云一致）
        colored_pcd_path = os.path.join(output_path, 'colored-cloud-points', f"{pcd_ts}.pcd")
        pcd_colored = o3d.geometry.PointCloud()
        pcd_colored.points = o3d.utility.Vector3dVector(points_lidar_np)
        pcd_colored.colors = o3d.utility.Vector3dVector(colored_points[:, :3])
        o3d.io.write_point_cloud(colored_pcd_path, pcd_colored)
        
        if debug_mode:
            print(f"  保存彩色点云: {colored_pcd_path}")
        
        # 保存灰度深度图（文件名与对应图像一致）
        def save_gray_depth(depth_map, cam_name):
            depth_np = depth_map.cpu().numpy()
            depth_np[depth_np == torch.inf] = 0  # 无效深度设为0
            normalized = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # 构建输出路径，使用原始图像的时间戳
            depth_img_path = os.path.join(output_path, f'depth-images/{cam_name}', f"{cam_ts}.png")
            cv2.imwrite(depth_img_path, normalized)
            
            if debug_mode:
                print(f"  保存{cam_name}深度图: {depth_img_path}")
        
        save_gray_depth(depth_map0, 'cam0')
        save_gray_depth(depth_map1, 'cam1')

if __name__ == "__main__":
    main()
