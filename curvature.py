import open3d as o3d
import numpy as np
from tqdm import tqdm

def pca_compute(data):
    average_data = np.mean(data, axis=0)
    decentration_matrix = data - average_data
    H = np.dot(decentration_matrix.T, decentration_matrix)
    eigenvalues, _ = np.linalg.eig(H)
    return np.sort(eigenvalues)[::-1]


def calculate_surface_curvature(file_path, output_file, radius=0.2, curvature_threshold=0.01):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    has_colors = pcd.has_colors()
    has_normals = pcd.has_normals()

    kdtree = o3d.geometry.KDTreeFlann(pcd)

    curvature = []
    sampling_points = []
    sampling_colors = []
    sampling_normals = []

    for i in tqdm(range(len(points)), desc="Calculating curvature"):
        k, indices, _ = kdtree.search_radius_vector_3d(points[i], radius)
        if k < 3:
            curvature.append(0)
            continue

        neighbors = points[indices, :]
        w = pca_compute(neighbors)
        delt = w[2] / np.sum(w) if np.sum(w) != 0 else 0
        curvature.append(delt)

        if delt >= curvature_threshold:
            sampling_points.append(points[i])
            if has_colors:
                sampling_colors.append(pcd.colors[i])
            if has_normals:
                sampling_normals.append(pcd.normals[i])

    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(np.array(sampling_points))
    if sampling_colors:
        sampled_pcd.colors = o3d.utility.Vector3dVector(np.array(sampling_colors))
    if sampling_normals:
        sampled_pcd.normals = o3d.utility.Vector3dVector(np.array(sampling_normals))

    o3d.io.write_point_cloud(output_file, sampled_pcd)
    return curvature

if __name__ == '__main__':
    points_path = r"..\voxel.pcd"
    output_file = r"..\curvature.pcd"

    curvature = calculate_surface_curvature(points_path, output_file, radius=0.2, curvature_threshold=0.08)




# import open3d as o3d
# import numpy as np
# from tqdm import tqdm
#
#
# def enhanced_curvature_calculation(neighbors):
#     """改进的曲率计算函数"""
#     # 计算协方差矩阵
#     cov_matrix = np.cov(neighbors.T)
#     eigenvalues = np.linalg.eigvalsh(cov_matrix)
#     eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列
#
#     # 改进曲率公式：增强平面边缘响应
#     if eigenvalues[0] < 1e-6:
#         return 0.0
#     curvature = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]  # 主曲率特征
#     return curvature
#
#
# def calculate_angle_variation(normals):
#     """计算法线方向变化强度"""
#     mean_normal = np.mean(normals, axis=0)
#     mean_normal /= np.linalg.norm(mean_normal)
#     angles = np.arccos(np.clip(np.dot(normals, mean_normal), -1.0, 1.0))
#     return np.std(angles)  # 使用标准差衡量方向变化
#
#
# def extract_building_contour(file_path, output_file,
#                              knn=50,
#                              curvature_threshold=0.15,
#                              angle_threshold=0.25,
#                              density_threshold=0.6):
#     # 数据预处理管线
#     pcd = o3d.io.read_point_cloud(file_path)
#
#     # 预处理步骤
#     pcd = pcd.voxel_down_sample(voxel_size=0.05)  # 降采样
#     pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)  # 去噪
#
#     # 法线估计与方向统一
#     if not pcd.has_normals():
#         pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
#     pcd.orient_normals_consistent_tangent_plane(k=30)
#
#     # 构建搜索结构
#     kdtree = o3d.geometry.KDTreeFlann(pcd)
#     points = np.asarray(pcd.points)
#     normals = np.asarray(pcd.normals)
#
#     contour_points = []
#     curvature_values = []
#     angle_variations = []
#     edge_colors = []
#     edge_normals = []
#
#     # 多特征并行计算
#     for idx in tqdm(range(len(points)), desc="特征分析"):
#         # KNN邻域搜索
#         k, indices, _ = kdtree.search_knn_vector_3d(pcd.points[idx], knn)
#         if k < 5:  # 确保足够邻域点
#             curvature_values.append(0)
#             angle_variations.append(0)
#             continue
#
#         # 获取邻域数据
#         neighbor_points = points[indices]
#         neighbor_normals = normals[indices]
#
#         # 计算三种特征
#         curvature = enhanced_curvature_calculation(neighbor_points)
#         angle_var = calculate_angle_variation(neighbor_normals)
#
#         # 邻域密度检测（防止边缘误判）
#         bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
#             o3d.utility.Vector3dVector(neighbor_points))
#         density = k / bbox.volume()
#
#         # 多条件联合判断
#         if (curvature > curvature_threshold and
#                 angle_var > angle_threshold and
#                 density > density_threshold):
#             contour_points.append(points[idx])
#             if pcd.has_colors():
#                 edge_colors.append(pcd.colors[idx])
#             if pcd.has_normals():
#                 edge_normals.append(normals[idx])
#
#         curvature_values.append(curvature)
#         angle_variations.append(angle_var)
#
#     # 后处理优化
#     contour_pcd = o3d.geometry.PointCloud()
#     contour_pcd.points = o3d.utility.Vector3dVector(np.array(contour_points))
#     if edge_colors:
#         contour_pcd.colors = o3d.utility.Vector3dVector(np.array(edge_colors))
#     if edge_normals:
#         contour_pcd.normals = o3d.utility.Vector3dVector(np.array(edge_normals))
#
#     # 密度聚类去噪
#     if len(contour_points) > 0:
#         labels = np.array(contour_pcd.cluster_dbscan(eps=0.1, min_points=10))
#         max_label = labels.max()
#         if max_label > 0:  # 排除噪声点（label=-1）
#             largest_cluster = np.argmax(np.bincount(labels[labels >= 0] + 1))
#             contour_pcd = contour_pcd.select_by_index(
#                 np.where(labels == largest_cluster - 1)[0])
#
#     # 保存结果
#     o3d.io.write_point_cloud(output_file, contour_pcd)
#
#     return curvature_values, angle_variations
#
#
# # 使用示例
# if __name__ == '__main__':
#     input_path = r"E:\\3D\\paper\\data\\Dortmund\\point cloud\\voxel.pcd"
#     output_path = r"E:\\3D\\paper\\data\\Dortmund\\point cloud\curvature.pcd"
#
#     # 建议参数范围：
#     # knn: 30-100 (根据点密度调整)
#     # curvature_threshold: 0.1-0.3 (需可视化调试)
#     # angle_threshold: 0.2-0.4 (弧度制，约11-23度)
#     # density_threshold: 0.5-1.0 (防止稀疏区域误判)
#     curvatures, angles = extract_building_contour(
#         input_path, output_path,
#         knn=60,
#         curvature_threshold=0.01,
#         angle_threshold=0.3,
#         density_threshold=0.7
#     )