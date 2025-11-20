import numpy as np
import open3d as o3d
import laspy


def read_pcd(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
    else:
        normals = None
        print("..")

    return points, colors, normals

def read_las(las_file):
    las = laspy.file.File(las_file, mode='r')
    points = np.array(las.points)
    r = las.Red
    g = las.Green
    b = las.Blue
    colors = []
    for i in range(len(r)):
        colors.append(np.array([r[i].astype('uint8'), g[i].astype('uint8'), b[i].astype('uint8')]))
    colors = np.array(colors)
    return points, colors

def GGLI(points_colors, gama=2.5):
    R, G, B = points_colors[0], points_colors[1], points_colors[2]
    ggli_top = 10 ** gama
    ggli_butt = ((2*G-R-B) / (2*G+R+B)).astype(np.complex128)
    ggli = ggli_top * ((ggli_butt ** gama)).astype(np.float64)
    return ggli

def NGRDI(points_colors):
    R, G, B = points_colors[0], points_colors[1], points_colors[2]
    NGRDI = (G - R) / (G + R)
    return NGRDI

def EXG(points_colors):
    R, G, B = points_colors[0], points_colors[1], points_colors[2]
    EXG = 2 * G - R - B
    return EXG

def VARI(points_colors):
    R, G, B = points_colors[0], points_colors[1], points_colors[2]
    VARI = (G - R) / (G + R - B)
    return VARI

def GLI(points_colors):
    R, G, B = points_colors[0], points_colors[1], points_colors[2]
    GLI = ((G-R) + (G-B)) / ((2*G)+R+B)
    return GLI

def RI(points_colors):
    R, G, B = points_colors[0], points_colors[1], points_colors[2]
    RI = R / G
    return RI

def EXG_EXR(points_colors):
    R, G, B = points_colors[0], points_colors[1], points_colors[2]
    ExG_ExR = (3*G-2.4*R-B)/(R+G+B)
    return ExG_ExR


def points_remove(pcd_file, save_file, index_func, threshold):
    points, colors, normals = read_pcd(pcd_file)

    indices = np.array([index_func(color) for color in colors])
    max_index = np.nanmax(indices)
    mask = indices <= (threshold * max_index)

    filtered_points = points[mask]
    filtered_colors = colors[mask]

    if normals is not None:
        filtered_normals = normals[mask]
    else:
        filtered_normals = None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors / 255.0)

    if filtered_normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(filtered_normals)

    o3d.io.write_point_cloud(save_file, pcd)

points_remove(r"..\voxel.pcd",
              r"..\GGLI.pcd",
              GGLI, 0.0005)