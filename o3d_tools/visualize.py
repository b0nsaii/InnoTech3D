import open3d as o3d
import numpy as np


class PointCloudProject:
    def __init__(self, project):
        self.project = project
        from o3d_tools.io import read_points, read_objects, read_masks, read_object_bb, get_global_mask
        # Read Point cloud file
        self.pcd = read_points(project)
        # Read objects bounding boxes as dict of pandas df's
        self.objects_df = read_objects(project)
        # Convert object bounding boxes into Open3d objects (for drawing purposes)
        self.objects = read_object_bb(self.objects_df)
        # Read object mask as dict
        self.masks = read_masks(project)
        # Convert to global mask (for drawing purposes)
        self.global_mask = get_global_mask(self.masks)


    def add_mask(self):
        # Read mask
        mask_indices = self.global_mask
        # Create a boolean mask
        mask = np.zeros(np.array(self.pcd.points).shape[0], dtype=bool)
        mask[mask_indices] = True
        # Get only mask subset of point cloud
        masked_points = np.array(self.pcd.points)[mask, :]
        masked_colors = np.array(self.pcd.colors)[mask, :]
        masked_colors = np.array([[0, 1, 0] for row in masked_colors])
        inv_masked_points = np.array(self.pcd.points)[~mask, :]
        inv_masked_colors = np.array(self.pcd.colors)[~mask, :]
        all_points = np.concatenate((masked_points, inv_masked_points))
        all_colors = np.concatenate((masked_colors, inv_masked_colors))
        masked_pcd = o3d.geometry.PointCloud()
        masked_pcd.points = o3d.utility.Vector3dVector(all_points)
        masked_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        return masked_pcd


    def draw_bb_only(self, which, plot=False, masked=False):
        if masked:
            pcd_masked = self.add_mask()
        cropped_pcd = []
        boxes = []
        for obj_type in which:
            for bb in self.objects[obj_type]:
                if masked:
                    cropped_pcd.append(pcd_masked.crop(bb))
                else:
                    cropped_pcd.append(self.pcd.crop(bb))
                boxes.append(bb)
        if plot:
            o3d.visualization.draw_geometries(cropped_pcd + boxes)
        return cropped_pcd


    def draw_bb_inverse(self, which_remove):
        cropped_pcd = self.draw_bb_only(which_remove)
        # Convert point clouds to numpy arrays
        original_points = np.asarray(self.pcd.points)
        original_colors = np.asarray(self.pcd.colors)
        remaining_points = np.copy(original_points)
        remaining_colors = np.copy(original_colors)
        for obj in cropped_pcd:
            cropped_points = np.asarray(obj.points)
            # Find the points that are not in the cropped points
            mask = np.isin(remaining_points, cropped_points, invert=True).all(axis=1)
            remaining_points = remaining_points[mask]
            remaining_colors = remaining_colors[mask]
        # Create a new point cloud with the remaining points
        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(remaining_points)
        remaining_pcd.colors = o3d.utility.Vector3dVector(remaining_colors)
        o3d.visualization.draw_geometries([remaining_pcd])
        return remaining_pcd
