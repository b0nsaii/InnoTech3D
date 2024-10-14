import pandas as pd
import open3d as o3d
import numpy as np

def read_points(project, voxel_size=None):
    # Columns contain x, y, z, i, r, g, b values
    xyz = pd.read_csv("data/TrainingSet/{}/{}.xyz".format(project, project),
                      sep=" ", usecols=range(0, 7), header=None)
    # Pass xyz and rgb to Open3D.o3d.geometry.PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz)[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.array(xyz)[:, 4:]/256)
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def read_objects(project):
    objects = pd.read_csv("data/TrainingSet/{}/{}.csv".format(project, project),
                          sep=",", header=0)
    object_dict = {}
    for label in objects[" Label"].unique():
        object_dict[label] = objects.loc[objects[" Label"] == label, :]
    return object_dict


def read_object_bb(object_dict):
    bounding_boxes = {}
    for object_type in object_dict.keys():
        bounding_boxes[object_type] = []
        for index, row in object_dict[object_type].iterrows():
            min_bound = np.array(row.iloc[2:5])
            max_bound = np.array(row.iloc[5:])
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            if object_type == 'Structural_IBeam':
                box.color = (1, 0, 0)
            elif object_type == 'HVAC_Duct':
                box.color = (0, 1, 0)
            elif object_type == 'Pipe':
                box.color = (0, 0, 1)
            elif object_type == 'Structural_ColumnBeam':
                box.color = (1, 0, 1)
            bounding_boxes[object_type].append(box)
    return bounding_boxes


def read_masks(project):
    import glob
    import os
    import re
    masks = {}
    folder = "data/TrainingSet/{}/{}.masks".format(project, project)
    files = glob.glob(os.path.join(folder, '*.txt'))
    for file in files:
        masks[re.search(r"(.*)mask\.txt", os.path.basename(file)).group(1)] = np.array(pd.read_csv(file, header=None))
    return masks


def get_global_mask(masks):
    return np.concatenate(list(masks.values()))
