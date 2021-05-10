import numpy as np
from torch.utils.data import Dataset
import torchvision
import os
import h5py
import pickle  # TODO or use h5py instead?
import trimesh

import config as cfg
import dataset.augmentation as Transforms

import sys
sys.path.append(cfg.BOP_PATH)
sys.path.append(os.path.join(cfg.BOP_PATH, "bop_toolkit_lib"))
import bop_toolkit_lib.inout as bop_inout
import bop_toolkit_lib.misc as bop_misc
import bop_toolkit_lib.dataset_params as bop_dataset_params

from glob import glob
import open3d as o3d
import pandas as pd
from pathlib import Path

class DatasetModelnet40(Dataset):

    """Adapted from RPM-Net (Yew et al., 2020): https://github.com/yewzijian/RPMNet"""

    def __init__(self, split, noise_type):
        dataset_path = cfg.M40_PATH
        categories = np.arange(20) if split in ["train", "val"] else np.arange(20, 40)
        split = "test" if split == "val" else split  # ModelNet40 has no validation set - use cat 0-19 with test set

        self.samples, self.labels = self.get_samples(dataset_path, split, categories)
        self.transforms = self.get_transforms(split, noise_type)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        sample = {'points': self.samples[item, :, :], 'label': self.labels[item], 'idx': np.array(item, dtype=np.int32)}

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_transforms(self, split, noise_type):
        # prepare augmentations
        if noise_type == "clean":
            # 1-1 correspondence for each point (resample first before splitting), no noise
            if split == "train":
                transforms = [Transforms.Resampler(1024),
                              Transforms.SplitSourceRef(),
                              Transforms.Scale(), Transforms.Shear(), Transforms.Mirror(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.ShufflePoints()]
            else:
                transforms = [Transforms.SetDeterministic(),
                              Transforms.FixedResampler(1024),
                              Transforms.SplitSourceRef(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.ShufflePoints()]
        elif noise_type == "jitter":
            # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
            if split == "train":
                transforms = [Transforms.SplitSourceRef(),
                              Transforms.Scale(), Transforms.Shear(), Transforms.Mirror(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.Resampler(1024),
                              Transforms.RandomJitter(),
                              Transforms.ShufflePoints()]
            else:
                transforms = [Transforms.SetDeterministic(),
                              Transforms.SplitSourceRef(),
                              Transforms.RandomTransformSE3_euler(),
                              Transforms.Resampler(1024),
                              Transforms.RandomJitter(),
                              Transforms.ShufflePoints()]
        else:
            raise ValueError(f"Noise type {noise_type} not supported for ModelNet40.")

        return torchvision.transforms.Compose(transforms)

    def get_samples(self, dataset_path, split, categories):
        filelist = [os.path.join(dataset_path, file.strip().split("/")[-1])
                   for file in open(os.path.join(dataset_path, f'{split}_files.txt'))]

        all_data = []
        all_labels = []
        for fi, fname in enumerate(filelist):
            f = h5py.File(fname, mode='r')
            data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
            labels = f['label'][:].flatten().astype(np.int64)

            if categories is not None:  # Filter out unwanted categories
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels


class DatasetScanObjectNN(Dataset):

    def __init__(self, split, noise_type):
        dataset_path = cfg.SON_PATH
        split = "test" if split == "val" else split

        self.samples, self.labels = self.get_samples(dataset_path, split)
        self.transforms = self.get_transforms(split, noise_type)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, item):
        sample = {'points': self.samples[item, :, :], 'label': self.labels[item], 'idx': np.array(item, dtype=np.int32)}

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_transforms(self, split, noise_type):
        # prepare augmentations
        if noise_type == "sensor" and split == "test":
            transforms = [Transforms.SetDeterministic(),
                          Transforms.SplitSourceRef(),
                          Transforms.RandomTransformSE3_euler(),
                          Transforms.Resampler(2048),
                          Transforms.ShufflePoints()]
        else:
            raise ValueError(f"Only noise type 'sensor' supported for SceneObjectNN.")

        return torchvision.transforms.Compose(transforms)

    def get_samples(self, dataset_path, split):
        filelist = [os.path.join(dataset_path, "test_objectdataset.h5")]

        all_data = []
        all_labels = []
        for fi, fname in enumerate(filelist):
            f = h5py.File(fname, mode='r')
            data = f['data'][:].astype(np.float32)
            labels = f['label'][:].flatten().astype(np.int64)

            all_data.append(data)
            all_labels.append(labels)
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels


class DatasetLinemod(Dataset):

    def __init__(self, split, noise_type):
        subsample = 16 if split == "eval" else 0  # only use every 16th test sample for evaluation during training
        split = "test" if split == "eval" else split
        self.samples, self.models = self.get_samples(split, subsample)
        self.transforms = self.get_transforms(split, noise_type)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        model = self.models[item['obj_id']]

        # compose sample
        sample = {
            'idx': idx,
            'points_src': item['pcd'],
            'points_ref': model,
            'scene': item['scene'],
            'frame': item['frame'],
            'cam': item['cam'],
            'gt': item['gt'],
        }
        if 'est' in item:  # initial estimate only given for test split (using PoseCNN)
            sample['est'] = item['est']

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_transforms(self, split, noise_type):
        # prepare augmentations
        if noise_type == "segmentation":
            if split == "train":
                transforms = [
                    # resample segmentation (with [p_fg]% from object)
                    Transforms.SegmentResampler(1024, p_fg=[0.5, 1.0]),
                    # align source and target using GT -- easier to define error this way
                    Transforms.GtTransformSE3(),
                    # normalize source and target (mean centered, max dist 1.0)
                    Transforms.Normalize(),
                    # apply an initial pose error
                    Transforms.RandomTransformSE3(rot_mag=90.0, trans_mag=1.0, random_mag=True)
                ]
            elif split == "val":
                transforms = [
                    Transforms.SetDeterministic(),
                    Transforms.SegmentResampler(1024, p_fg=[0.5, 1.0]),
                    Transforms.GtTransformSE3(),
                    Transforms.Normalize(),
                    Transforms.RandomTransformSE3(rot_mag=90.0, trans_mag=1.0, random_mag=True)
                ]
            else:  # start from posecnn
                transforms = [
                    Transforms.SetDeterministic(),
                    # randomly resample inside segmentation mask (estimated by PoseCNN)
                    Transforms.SegmentResampler(1024, p_fg=1.0, patch=False),
                    # initial (erroneous) alignment using PoseCNN's pose estimation
                    Transforms.EstTransformSE3(),
                    Transforms.Normalize()
                ]
        else:
            raise ValueError(f"Noise type {noise_type} not supported for LINEMOD.")
        return torchvision.transforms.Compose(transforms)

    def get_samples(self, split, subsample=0):
        model_params = bop_dataset_params.get_model_params('/'.join(cfg.LM_PATH.split('/')[:-1]),
                                                           cfg.LM_PATH.split('/')[-1], 'eval')
        mesh_ids = model_params['obj_ids']

        models = dict()
        for mesh_id in mesh_ids:
            mesh = trimesh.load(os.path.join(cfg.LM_PATH, f"models_eval/obj_{mesh_id:06d}.ply"))
            pcd, face_indices = trimesh.sample.sample_surface_even(mesh, 4096)
            models[mesh_id] = np.hstack([pcd, mesh.face_normals[face_indices]]).astype(np.float32)

        samples_path = f"reagent/{split}_posecnn.pkl" if split == "test" else f"reagent/{split}.pkl"
        with open(os.path.join(cfg.LM_PATH, samples_path), 'rb') as file:
            samples = pickle.load(file)
        if subsample > 0:  # used for evaluation during training
            samples = samples[::subsample]
        return samples, models

    # for visualization
    def get_rgb(self, scene_id, im_id):
        dataset_path = os.path.join(cfg.LM_PATH, "test")
        scene_path = os.path.join(dataset_path, f"{scene_id:06d}")
        file_path = os.path.join(scene_path, f"rgb/{im_id:06d}.png")
        if os.path.exists(file_path):
            return bop_inout.load_im(file_path)[..., :3]/255
        else:
            print(f"missing file: {file_path}")
            return np.zeros((480, 640, 3), dtype=np.float32)

    def get_depth(self, scene_id, im_id):
        dataset_path = os.path.join(cfg.LM_PATH, "test")
        scene_path = os.path.join(dataset_path, f"{scene_id:06d}")
        file_path = os.path.join(scene_path, f"depth/{im_id:06d}.png")
        if os.path.exists(file_path):
            return bop_inout.load_depth(file_path)
        else:
            print(f"missing file: {file_path}")
            return np.zeros((480, 640), dtype=np.float32)

    def get_seg(self, scene_id, im_id, gt_id):
        dataset_path = os.path.join(cfg.LM_PATH, "test")
        scene_path = os.path.join(dataset_path, f"{scene_id:06d}")
        file_path = os.path.join(scene_path, f"mask_visib/{im_id:06d}_{gt_id:06d}.png")
        if os.path.exists(file_path):
            return bop_inout.load_im(file_path)
        else:
            print(f"missing file: {file_path}")
            return np.zeros((480, 640), dtype=np.uint8)

import copy
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

class DatasetSevenScenes(Dataset):

    def __init__(self, split, noise_type, subsample= 0):
        """Initialize the 7-scenes RedKitchen Dataset

        Args:
            split ([string]): train, test or val
            noise_type ([type]): only clean is allowed, as the data is already
            noisy
            subsample (int, optional): [Choose which views are used for
            train, test and val]. Use the subsample-th views for training, the
            views between for validation and the rest for testing. Defaults to
            -1, that uses seq-01 for training, seq-02 for validation and the
            rest for testing
        """

        split = "test" if split == "val" else split # 7-scenes has no validation set

        # camera intrinsics
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsics.set_intrinsics(640,480,585,585,320,240)

        # load map pointcloud if exists, otherwise reconstruct
        self.scene = self.get_scene_pointcloud(cfg.SEVEN_SCENES_PATH)

        # load from rgbd images
        self.samples = self.get_samples(split, subsample)

        self.transforms = self.get_transforms(split, noise_type)

    @classmethod
    def read_rgbdpose_frame(cls, frame_name):
        color_file = frame_name + ".color.png"
        depth_file = frame_name + ".depth.png"
        poseFile = f"{frame_name}.pose.txt"
        if not os.path.exists(color_file):
            print(f"No rgb information found at \"{color_file}\"")
            exit(-3)
        if not os.path.exists(depth_file):
            print(f"No depth information found at \"{depth_file}\"")
            exit(-3)
        if not os.path.exists(poseFile):
            print(f"No pose information found for \"{frame_name}\" at \"{poseFile}\"")
            exit(-3)
        color = o3d.io.read_image(color_file)
        depth = o3d.io.read_image(depth_file)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
            )
        pose = pd.read_csv(poseFile, '\t', dtype=float, header=None, usecols=list(range(4)))
        pose = pose.to_numpy()
        return rgbd_image, pose

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_transforms(self, split, noise_type):
        # prepare augmentations
        if noise_type == "clean":
            if split == 'train':
                transforms = [
                    Transforms.SetDeterministic(),
                    # apply an initial pose error to world
                    Transforms.RandomTransformSE3(rot_mag=45, trans_mag=1.0, random_mag=True, use_source=False),
                    # Frustum culling
                    Transforms.FrustumCulling(self.intrinsics, 1, 45),
                    # subsampling
                    Transforms.Resampler(2048)
                ]
            else:
                transforms = [
                    # apply an initial pose error to world
                    Transforms.RandomTransformSE3(rot_mag=25, trans_mag=.5, random_mag=True, use_source=False),
                    # Frustum culling
                    Transforms.FrustumCulling(self.intrinsics, 1, 45),
                    # subsampling
                    Transforms.Resampler(2048)
                ]
        else:
            raise ValueError(f"Noise type {noise_type} not supported for SevenScenes.")
        return torchvision.transforms.Compose(transforms)

    @staticmethod
    def show_sample(sample):
        target = o3d.geometry.PointCloud()
        source = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(sample['points_ref'])
        source.points = o3d.utility.Vector3dVector(sample['points_src'])

        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([source_temp, target_temp])

    @staticmethod
    def show_pointcloud_pair(source_array, target_array, transform):
        target = o3d.geometry.PointCloud()
        source = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_array)
        source.points = o3d.utility.Vector3dVector(source_array)

        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transform)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def get_samples(self, split, subsample = 0):
        dataset_path = cfg.SEVEN_SCENES_PATH
        scene_name = os.path.basename(dataset_path)
        frame_names = self.get_framenames(dataset_path, split, subsample)

        # prepare sampling transformation
        subsampler = Transforms.Resampler(4096, source_only = True)
        # prepare alginment transformation
        aligner = Transforms.GtTransformSE3(source_to_target=False)

        static_transforms = torchvision.transforms.Compose([aligner, subsampler])

        samples = []

        for i,frame_name in enumerate(frame_names):
            # read view pointclouds
            rgbdImage, pose = DatasetSevenScenes.read_rgbdpose_frame(frame_name)
            viewPointCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbdImage, self.intrinsics)
            viewPointCloud = np.asarray(viewPointCloud.points)

            pose = np.linalg.inv(pose)

            sequence_name = Path(frame_name).parent.name

            sample = {
                'idx': i,
                'label' : i,
                'points_src': viewPointCloud,
                'points_ref': self.scene,
                'scene': scene_name,
                'sequence' : sequence_name,
                'frame': os.path.basename(frame_name),
                'gt': {
                    'cam_R_m2c': pose[np.ix_(range(3),range(3))],
                    'cam_t_m2c': pose[np.ix_(range(3),[3])]
                },
                'cam' : {
                    'cam_K' : self.intrinsics.intrinsic_matrix,
                    'obj_id' : scene_name
                }
            }

            sample = static_transforms(sample)
            samples.append(sample)

        return samples

    def get_scene_pointcloud(self, dataset_path):
        if os.path.exists(os.path.join(dataset_path, "integrated_mesh.ply")):
            # read world map from ply
            mesh = o3d.io.read_triangle_mesh(os.path.join(dataset_path, "integrated_mesh.ply"))
        else:
            # reconstruct
            filenames = glob(os.path.join(dataset_path, "seq-[0-9][0-9]", "*.color.png"))
            filenames = [f[:-10] for f in filenames]
            filenames.sort()


            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length = 4/512,
                sdf_trunc = 0.04,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                    )
            intrinsics = o3d.camera.PinholeCameraIntrinsic()
            intrinsics.set_intrinsics(640,480,585,585,320,240)


            for frame_name in filenames:
                print(f"Integrating {frame_name}")
                rgbdImage, pose = DatasetSevenScenes.read_rgbdpose_frame(frame_name)
                pose = np.linalg.inv(pose)
                volume.integrate(rgbdImage, intrinsics, pose)

            mesh = volume.extract_triangle_mesh()
            mesh = mesh.simplify_quadric_decimation(100000)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            o3d.io.write_triangle_mesh(os.path.join(dataset_path, "integrated_mesh.ply"), mesh)

        return np.asarray(mesh.vertices)


    def get_framenames(self, dataset_path, split, subsample = 0):
        frames = []
        if subsample <= 0:
            if subsample == 0:
                # use trainsplit from dataset
                with open(os.path.join(dataset_path, "TrainSplit.txt")) as train_split_file:
                    train_names = train_split_file.read().split("\n")
                    train_paths = []
                    for train_name in train_names:
                        if len(train_name) == 0:
                            continue
                        train_name = train_name[8:]
                        train_num = int(train_name)
                        train_path = os.path.join(dataset_path, f"seq-{train_num:02d}")
                        train_paths.append(train_path)
                    #train_paths = [os.path.join(dataset_path, train_path.replace("sequence","seq-")) for train_path in train_paths if len(train_path) > 0]
            else:
                # use single sequence -subsample
                train_paths = [os.path.join(dataset_path, f"seq{subsample:03d}")]
            sequence_paths = glob(os.path.join(dataset_path, "seq-[0-9][0-9]"))
            if not all(train_path in sequence_paths for train_path in train_paths):
                raise ValueError(f"A sequence does not exist in {dataset_path}")
            if split == "train":
                split_paths = train_paths
            else:
                split_paths = [path for path in sequence_paths if path not in train_paths]
            split_paths.sort()
            for split_path in split_paths:
                seq_frames = glob(os.path.join(split_path, "*.color.png"))
                seq_frames.sort()
                frames.extend(seq_frames)
            # cut of .color.png suffix
            frames = [f[:-10] for f in frames]
        else:
            # use every subsamplte-th for training, rest for testing
            if subsample == 1:
                raise ValueError(f"Cannot use all frames for training. Use subsample != 1")
            frames = glob(os.path.join(dataset_path, "seq-[0-9][0-9]", "*.color.png"))
            frames.sort()
            if split == "train":
                frames = [f for i,f in enumerate(frames) if i%subsample == 0]
            else:
                frames = [f for i,f in enumerate(frames) if i%subsample != 0]
        return frames

