import cv2
import numpy as np
import os
import open3d as o3d
import pandas as pd
import math
from shapely.geometry import Point, MultiPoint
from shapely.geometry import Point, LineString
from shapely.strtree import  STRtree
import matplotlib.pyplot as plt
import pyproj
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from shapely.ops import nearest_points
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def o3dread_pcd(pcd_file, camera_position=None):
    pcd = o3d.io.read_point_cloud(pcd_file)

    if not pcd.has_normals():
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = None
    else:
        pcd.orient_normals_consistent_tangent_plane(5)  # 最小生成树
        normals = np.asarray(pcd.normals)

    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    o = np.ones((points.shape[0], 1), dtype=np.float32)
    points = np.hstack((points, o))

    return points, colors, normals

def read_pcd_from_txt(txt_file):
    data = np.loadtxt(txt_file)
    points = data[:, :3]
    colors = data[:, 3:7]

    o = np.ones((points.shape[0], 1))

    points = np.hstack((points, o, colors)).astype(np.float32)

    return points

def project(points_file, image, M1, M2, M3, camera_position):
    coords_3D, colors, normals = o3dread_pcd(points_file, camera_position=camera_position)
    resolution = image.shape

    coords_3D[:, 0] = coords_3D[:, 0] - 2438000
    coords_3D[:, 1] = coords_3D[:, 1] + 5038000
    coords_3D[:, 2] = coords_3D[:, 2] + 3047000

    ext_transform = np.matmul(M1, M2)
    extrinsic = np.linalg.inv(ext_transform)
    coords_1 = np.matmul(M3, extrinsic)

    coords_2D = coords_3D @ coords_1.T

    indices = np.arange(coords_2D.shape[0])
    coords_2D = np.column_stack((coords_2D, indices))
    coords_2D = np.column_stack((coords_2D, colors))

    coords_2D = coords_2D[np.where(coords_2D[:, 2] > 0)]
    coords_2D[:, 2] = np.clip(coords_2D[:, 2], a_min=1e-5, a_max=1e5)
    coords_2D[:, 0] /= coords_2D[:, 2]
    coords_2D[:, 1] /= coords_2D[:, 2]
    coords_2D = coords_2D[np.where(coords_2D[:, 0] > 0)]
    coords_2D = coords_2D[np.where(coords_2D[:, 0] < resolution[1])]
    coords_2D = coords_2D[np.where(coords_2D[:, 1] > 0)]
    coords_2D = coords_2D[np.where(coords_2D[:, 1] < resolution[0])]

    return coords_2D, colors

def show_with_opencv(image, save_img, coords, points_color):
    canvas = image.copy()
    cv2.putText(canvas,
                text='',
                org=(90, 180),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=12.0,
                thickness=10,
                color=(0, 0, 255))
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    if coords is not None:
        for index in range(len(coords)):
            if coords[index][0] == None or coords[index][1] == None:
                p = (0, 0)
            else:
                p = (int(coords[index][0]), int(coords[index][1]))

            cv2.circle(canvas, p, 2, color=[255, 0, 0], thickness=5)
    canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_img, canvas)

def camera_params(params_file, number):
    params = pd.DataFrame(pd.read_excel(params_file))

    if number == 0:
        cam_intrinsics = np.array([[params['focal_length'][number], 0, 2985.9336103877236, 0],
                               [0, params['focal_length'][number], 2070.0098461126446, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)
    else:
        cam_intrinsics = np.array([[params['focal_length'][number], 0, 2985.9336103877236, 0],
                               [0, params['focal_length'][number], 2068.5098461126446, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=np.float32)
    chunk_transform = np.array(
        [[params['scale'][number] * params['chunk_R11'][number], params['scale'][number] * params['chunk_R12'][number],
          params['scale'][number] * params['chunk_R13'][number], params['chunk_T1'][number]],
         [params['scale'][number] * params['chunk_R21'][number], params['scale'][number] * params['chunk_R22'][number],
          params['scale'][number] * params['chunk_R23'][number], params['chunk_T2'][number]],
         [params['scale'][number] * params['chunk_R31'][number], params['scale'][number] * params['chunk_R32'][number],
          params['scale'][number] * params['chunk_R33'][number], params['chunk_T3'][number]],
         [0, 0, 0, 1]], dtype=np.float32)

    cam_extrinsic = np.array([[params['camera11'][number], params['camera12'][number], params['camera13'][number],
                               params['camera14'][number]],
                              [params['camera21'][number], params['camera22'][number], params['camera23'][number],
                               params['camera24'][number]],
                              [params['camera31'][number], params['camera32'][number], params['camera33'][number],
                               params['camera34'][number]],
                              [0, 0, 0, 1]], dtype=np.float32)

    cam_position = np.array([params['Tx'][number], params['Ty'][number], params['Tz'][number]],
                         dtype=np.float32)
    camera_position = cam_position.reshape(1, 3)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    camera_coordinates = transformer.transform(camera_position[:, 0], camera_position[:, 1], camera_position[:, 2])
    camera_coordinates_array = np.column_stack(camera_coordinates)

    return chunk_transform, cam_extrinsic, cam_intrinsics, camera_coordinates_array

def plot_epipolar_line_with_point(img, epipolar_line, img_save_path, random_color, line_label, point, number):
    A, B, C = epipolar_line

    img_height, img_width = img.shape[:2]

    x_top = -(C) / A if A != 0 else None
    y_top = 0

    x_bottom = -(C + B * img_height) / A if A != 0 else None
    y_bottom = img_height

    x_left = 0
    y_left = -(C + A * x_left) / B if B != 0 else None
    x_right = img_width
    y_right = -(C + A * x_right) / B if B != 0 else None

    points = []
    if x_top is not None and 0 <= x_top <= img_width:
        points.append((x_top, y_top))
    if x_bottom is not None and 0 <= x_bottom <= img_width:
        points.append((x_bottom, y_bottom))
    if y_left is not None and 0 <= y_left <= img_height:
        points.append((x_left, y_left))
    if y_right is not None and 0 <= y_right <= img_height:
        points.append((x_right, y_right))

    if len(points) >= 2:
        points = sorted(points, key=lambda x: x[0])
        x_vals = [point[0] for point in points]
        y_vals = [point[1] for point in points]

        random_color_normalized = tuple(c / 255 for c in random_color)
        plt.imshow(img, cmap='gray')
        plt.plot(x_vals, y_vals, color=random_color_normalized, label="Epipolar Line", linewidth=1)

        text_x = (x_vals[0] + x_vals[1]) / 2
        text_y = (y_vals[0] + y_vals[1]) / 2
        plt.text(text_x, text_y, line_label, color='yellow', fontsize=5, ha='center', va='center')

        if number == 0:
            plt.scatter(point[0], point[1], color='green', label='Point', s=1.5)
        else:
            plt.scatter(point[0], point[1], color='red', label='Point', s=1.5)

        plt.axis('off')
        plt.tight_layout()

        plt.savefig(img_save_path, dpi=600, bbox_inches='tight')

def distance_from_point_to_line(point, line):
    x, y = point
    k, b = line['k'], line['b']

    A = -k
    B = 1
    C = -b

    distance = abs(A * x + B * y + C) / np.sqrt(A ** 2 + B ** 2)

    x_nearest = x - (k * (x - 0) + (y - b)) / (1 + k ** 2)
    y_nearest = k * x_nearest + b
    return distance, (x_nearest, y_nearest)


def check_point_in_line_regions(point, regSave, linesInfo):

    lines_in_region = []
    min_distance = float('inf')
    closest_line_info = None
    closest_point = None

    for i, reg in enumerate(regSave):
        x_min, y_min = np.min(reg, axis=0)
        x_max, y_max = np.max(reg, axis=0)

        x, y = point
        if x_min <= x <= x_max and y_min <= y <= y_max:
            lines_in_region.append(linesInfo[i])

            line = linesInfo[i]
            distance, nearest_point = distance_from_point_to_line(point, line)

            if distance < min_distance:
                min_distance = distance
                closest_line_info = line
                closest_point = nearest_point

    return lines_in_region, closest_line_info, closest_point

def calculate_intersection(A1, B1, C1, A2, B2, C2):
    denominator = A1 * B2 - A2 * B1
    if denominator == 0:
        return None, None
    x = (B2 * C1 - B1 * C2) / denominator
    y = (A1 * C2 - A2 * C1) / denominator
    return x, y

def calculate_intersec(line, A, B, C):
    if B != 0:
        x1, y1 = -10000, -(A * (-10000) + C) / B
        x2, y2 = 10000, -(A * 10000 + C) / B
    elif A != 0:
        x1, y1 = -C / A, -10000
        x2, y2 = -C / A, 10000
    else:
        raise ValueError("A and B =/0")

    line2 = LineString([(x1, y1), (x2, y2)])
    intersection = line.intersection(line2)

    if intersection.is_empty:
        return None, None
    else:
        return (intersection.x, intersection.y)

def read_lines_from_excel(file_path):
    df = pd.read_excel(file_path)
    line = []

    for _, row in df.iterrows():
        x1 = row['x1']
        y1 = row['y1']
        x2 = row['x2']
        y2 = row['y2']

        dx = x2 - x1
        dy = y2 - y1
        line.append([x1, y1, x2, y2])

    return line


def draw_lines_on_image(image_path, line, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_path, cmap='gray')

    for l in line:
        x1, y1, x2, y2 = l.coords[0][0], l.coords[0][1], l.coords[1][0], l.coords[1][1]
        plt.plot([x1, x2], [y1, y2], color='red', linewidth=0.5)

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def draw_lines_on_image1(image_path, line, output_path, point):

    plt.figure(figsize=(10, 10))
    plt.imshow(image_path, cmap='gray')

    x1, y1, x2, y2 = line.coords[0][0], line.coords[0][1], line.coords[1][0], line.coords[1][1]
    plt.plot([x1, x2], [y1, y2], color='red', linewidth=0.5)

    plt.scatter(point[0], point[1], color='green', label='Point', s=2)  # s为点的大小

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def remove_duplicates(input_list):
    seen = set()
    result = []

    for sublist in input_list:
        key = (sublist[0], sublist[1])
        if key not in seen:
            seen.add(key)
            result.append(sublist)

    return result

def cloud_restraint(lines, coords, restrainted_points, line_img_path, imgs, epipolar_line_path,
                    initial_buffer_size, max_buffer_size, buffer_step,
                    img_number, fundamental_matrix, sample_num=10):

    img_gray = cv2.cvtColor(line_img_path, cv2.COLOR_BGR2GRAY)
    retval2, img_pro = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    lines = read_lines_from_excel(lines)
    # all_lines = [LineString([(x1, y1), (x2, y2)]) for k, b, dx, dy, x1, y1, x2, y2, length in lines]
    all_lines = [LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in lines]

    lines_in_image = []
    for line in all_lines:
        is_valid = True
        for i in range(sample_num + 1):
            t = i / sample_num
            x = line.interpolate(t).x
            y = line.interpolate(t).y
            if not (0 <= x < img_pro.shape[1] and 0 <= y < img_pro.shape[0]):
                is_valid = False
                break
            if img_pro[int(y), int(x)] != 255:
                is_valid = False
                break
        if is_valid:
            lines_in_image.append(line)

    restrainted_dict = {}
    if img_number > 0:
        for p in restrainted_points[img_number-1]:
            restrainted_dict[p[2]] = p

    restraints_point = []

    for index, coord in enumerate(tqdm(coords, desc="Processing Points")):
        point = Point(float(coord[0]), float(coord[1]))
        buffer_size = initial_buffer_size

        potential_lines = False
        while buffer_size <= max_buffer_size:
            buffer = point.buffer(buffer_size, resolution=32)
            potential_lines = [line for line in all_lines if buffer.intersects(line)]
            buffer_size += buffer_step

        if potential_lines:
            # filtered_lines = [
            #     line for line in potential_lines
            #     if line.length >= 45
            # ]
            distances = [point.distance(line) for line in potential_lines]
            if len(distances) == 1:
                nearest_line = potential_lines[0]
            else:
                nearest_line = potential_lines[np.argmin(distances)]

            (x1, y1), (x2, y2) = nearest_line.coords[0], nearest_line.coords[1]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2

            if img_number == 0:
                closest_point = nearest_points(point, nearest_line)[1]
                restraints_point.append([closest_point.x, closest_point.y, coord[4], 1, coord[5], coord[6], coord[7]])

            else:
                closest_point = nearest_points(point, nearest_line)[1]
                restraints_point.append([closest_point.x, closest_point.y, coord[4], 1, coord[5], coord[6], coord[7]])

        else:
            if img_number == 0:
                restraints_point.append([coord[0], coord[1], coord[4], 0, coord[5], coord[6], coord[7]])
            else:
                restraints_point.append([coord[0], coord[1], coord[4], 0, coord[5], coord[6], coord[7]])

    return restraints_point

def Calculate_fundamental_matrix(coords, total_coords):
    homonous_points = []
    for coord in coords:
        x, y, index= coord[0], coord[1], coord[4]
        homongenuous_indice = total_coords[0][:, 4]
        if index in homongenuous_indice.flatten():
            dice = np.where(homongenuous_indice == index)[0][0]
            homongenuous_point = total_coords[0][dice]
            homonous_points.append([coord, homongenuous_point])
        if len(homonous_points) == 100:
            break
    A = []
    b = []

    for list in homonous_points:
        x1, y1 = list[0][0], list[0][1]
        x2, y2 = list[1][0], list[1][1]
        A.append([x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1])
        b.append(0)

    A = np.array(A)

    U, S, Vt = np.linalg.svd(A)
    f_values = Vt[-1, :]

    fundamental_matrix = np.reshape(f_values, (3, 3))

    U, S, Vt = np.linalg.svd(fundamental_matrix)
    S[2] = 0
    fundamental_matrix = U @ np.diag(S) @ Vt

    return fundamental_matrix


def save_projection(projection_txt, restraints_point):
    data_lines = []
    for line in restraints_point:
        data_lines.append(' '.join(map(str, line)))

    with open(projection_txt, 'w') as save_txt:
        save_txt.writelines('\n'.join(data_lines) + '\n')


def main(dense_points_path, cam_params_path, images_path, line_excel, projection_img_savepath,
         cloud_point_restraints_save_path, epipolar_line_save_path, restrainted_homonyous_points_path):
    restrainted_points = []
    total_coords = []
    imgs = []
    for i, name in enumerate(os.listdir(images_path)):
        print(name)

        img_line_name = os.path.join(images_path, name)

        line_path = os.path.join(line_excel, name.split('.')[0] + '.xlsx')

        projection_img = os.path.join(projection_img_savepath, name)
        cloud_point_restraint_img = os.path.join(cloud_point_restraints_save_path, name)
        epipolar_line_img_path = os.path.join(epipolar_line_save_path, name)

        projection_txt = os.path.join(restrainted_homonyous_points_path, name.split('.')[0] + '.txt')
        img_line = cv2.imread(img_line_name)

        chunk_transform, cam_extrinsic, cam_intrinsics, camera_position = camera_params(cam_params_path, i)
        coords, points_color = project(dense_points_path, img_line, chunk_transform, cam_extrinsic, cam_intrinsics,
                                       camera_position)
        show_with_opencv(img_line, projection_img, coords=coords, points_color=points_color)

        total_coords.append(coords)
        if i == 0:
            fundamental_matrix = None
        else:
            fundamental_matrix = Calculate_fundamental_matrix(coords, total_coords)

        imgs.append(img_line)
        restraints_point = cloud_restraint(line_path, coords, restrainted_points, img_line, imgs,
                                           epipolar_line_img_path, initial_buffer_size=10, max_buffer_size=50,
                                           buffer_step=5, img_number=i, fundamental_matrix=fundamental_matrix)

        show_with_opencv(img_line, cloud_point_restraint_img, coords=restraints_point, points_color=points_color)

        restraints_point = remove_duplicates(restraints_point)
        restrainted_points.append(restraints_point)

        save_projection(projection_txt, coords)

dense_points_path = r'..\positive point cloud.pcd'
cam_params_path = r'..\positive point cloud-two images.xlsx'
images_path = r'..\line detection result'
line_path = r'..\original lines'
projection_img_savepath = r'..projection to images'
cloud_point_restraints_save_path = r'..\optimized point cloud'
epipolar_line_save_path = r'..\epipolar line map'
restrainted_homonyous_points_path = r'..\homonyous points'

main(dense_points_path, cam_params_path, images_path, line_path, projection_img_savepath,
     cloud_point_restraints_save_path, epipolar_line_save_path, restrainted_homonyous_points_path)