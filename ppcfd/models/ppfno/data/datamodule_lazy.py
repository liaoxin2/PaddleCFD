import copy
import json
import logging
import os
import random
import re
import sys
import unittest
import warnings
from collections.abc import Callable
from pathlib import Path
from timeit import default_timer
from typing import Dict, List, Optional, Tuple, Union

import meshio
import numpy as np
import open3d as o3d
import paddle
import math

from .base_datamodule import BaseDataModule

from ..neuralop.utils import UnitGaussianNormalizer


def get_last_dir(path):
    return os.path.basename(os.path.normpath(path))


def compute_q_ref(info):
    """计算样本参考动压 q_ref = 0.5 * rho * flow_speed**2。

    flow_speed 为合成来流速度，与 F_const/M_const 使用完全相同的定义，
    保证 F_const == 1/(q_ref * reference_area)。

    注意：info["car_speed"] 在数据加载时已统一转换为 m/s（/3.6），
    此函数假定传入的 car_speed 已是 m/s。
    """
    mass_density = float(info["density"])
    car_speed = float(info["car_speed"])  # m/s
    wind_speed = float(info["wind_speed"])
    wind_angle_rad = math.radians(float(info["wind_angle"]))
    vx = car_speed + wind_speed * math.cos(wind_angle_rad)
    vy = wind_speed * math.sin(wind_angle_rad)
    flow_speed = math.sqrt(vx**2 + vy**2)
    q_ref = 0.5 * mass_density * flow_speed**2
    return q_ref


class LoadMesh:

    def __init__(self, path, query_points=None, closest_points_to_query=False):
        self.path = path
        self.query_points = query_points
        self.closest_points_to_query = closest_points_to_query

    def index_to_mesh_path(self, index, extension: str = ".ply") -> Path:
        return self.path / ("mesh_" + index + extension)

    def load_mesh(self, mesh_path: Path) -> o3d.geometry.TriangleMesh:
        assert mesh_path.exists(), "Mesh path does not exist"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        return mesh

    def load_mesh_tri(self, mesh_path: Path) -> o3d.t.geometry.TriangleMesh:
        mesh = self.load_mesh(mesh_path)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        return mesh

    def vertices_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> paddle.Tensor:
        return paddle.to_tensor(data=np.asarray(mesh.vertices).astype(np.float32))

    def triangles_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> paddle.Tensor:
        return paddle.to_tensor(data=np.asarray(mesh.triangles).astype(np.int64))

    def get_triangle_centroids(
        self, vertices: paddle.Tensor, triangles: paddle.Tensor
    ) -> paddle.Tensor:
        A, B, C = (
            vertices[triangles[:, 0]],
            vertices[triangles[:, 1]],
            vertices[triangles[:, 2]],
        )
        centroids = (A + B + C) / 3
        areas = (
            paddle.sqrt(x=paddle.sum(x=paddle.cross(x=B - A, y=C - A) ** 2, axis=1)) / 2
        )
        return centroids, areas

    def compute_df(self, mesh: o3d.t.geometry.TriangleMesh) -> np.ndarray:
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        distance = scene.compute_distance(self.query_points).numpy()
        if self.closest_points_to_query:
            closest_points = scene.compute_closest_points(self.query_points)[
                "points"
            ].numpy()
        else:
            closest_points = None
        return distance, closest_points

    def save_df_closest(self, df_closest_dict, index: str, file_type="npy"):
        if file_type == ".pdparams":
            for k, v in df_closest_dict.items():
                df_closest_dict[k] = paddle.to_tensor(data=v)
            paddle.save(
                obj=df_closest_dict,
                path=os.path.join(
                    os.path.dirname(self.path), f"df_closest_{index}.pdparams"
                ),
            )
        elif file_type == "npy":
            np.save(
                os.path.join(os.path.dirname(self.path), f"df/df_{index}.npy"),
                df_closest_dict["df"],
            )
            np.save(
                os.path.join(os.path.dirname(self.path), f"df/closest_{index}.npy"),
                df_closest_dict["closest"],
            )

    def get_df_closest(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh], index: str = None
    ):
        assert self.query_points is not None, "query_points does not be None"
        if isinstance(mesh, Path):
            mesh = self.load_mesh_tri(mesh)
        df, closest = self.compute_df(mesh)
        df_closest_dict = {"df": df, "closest": closest}
        if index is not None:
            self.save_df_closest(df_closest_dict, index)
        return paddle.to_tensor(data=df), paddle.to_tensor(data=closest)

    def compute_sdf(self, mesh: Union[Path, o3d.t.geometry.TriangleMesh]) -> np.ndarray:
        if isinstance(mesh, Path):
            mesh = self.load_mesh(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        signed_distance = scene.compute_signed_distance(self.query_points).numpy()
        if self.closest_points_to_query:
            closest_points = scene.compute_closest_points(self.query_points)[
                "points"
            ].numpy()
        else:
            closest_points = None
        return signed_distance, closest_points

    def sdf_vertices_closest_from_mesh(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh]
    ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        assert self.query_points is not None, "query_points does not be None"
        if isinstance(mesh, Path):
            mesh = self.load_mesh_tri(mesh)
        sdf, closest_points = self.compute_sdf(mesh)
        vertices = mesh.vertex.positions.numpy()
        return (
            paddle.to_tensor(data=sdf),
            vertices,
            paddle.to_tensor(data=closest_points),
        )


class LoadFile:

    def __init__(self, path):
        self.path = path

    def index_to_file(self, filename: str, extension: str = ".npy") -> Path:
        return self.path / (filename + extension)

    def load_file(
        self, file_path: Union[Path, str], extension: str = ".npy"
    ) -> paddle.Tensor:
        if isinstance(file_path, str):
            file_path = self.index_to_file(file_path, extension)
        assert file_path.exists(), f"File path {file_path} does not exist"
        if extension == ".npy":
            float_max = np.finfo(np.float32).max
            float_min = np.finfo(np.float32).min
            data_double = np.load(str(file_path))
            data_clipped = np.clip(data_double, float_min, float_max * 1e-20)
            assert not np.isinf(data_clipped).any(), "存在溢出值！"
            assert not np.isnan(data_clipped).any(), "存在无效值！"
            data = paddle.to_tensor(data_clipped.astype(np.float32))

        elif extension == ".pdparams":
            data = paddle.load(path=str(str(file_path)))
        return data


class PathDictDataset(paddle.io.Dataset, LoadMesh, LoadFile):

    def __init__(
        self,
        path: str = None,
        query_points=None,
        closest_points_to_query=False,
        indices: Optional[List[str]] = None,
        norms_dict: Optional[Dict[str, Callable]] = {},
        data_keys: Optional[List[str]] = ["info", "pressure", "wallshearstress"],
        out_keys: Optional[List[str]] = ["pressure", "wallshearstress"],
        lazy_loading=True,
    ):
        LoadMesh.__init__(self, path, query_points, closest_points_to_query)
        LoadFile.__init__(self, path)
        assert path is not None, "path is None"
        self.path = Path(path)
        self.indices = indices
        self.norms_dict = norms_dict
        self.data_keys = data_keys
        # 输出物理场的 key（pressure/wallshearstress），这些 key 需要按 q_ref 做系数归一化
        self.out_keys = out_keys
        self.lazy_loading = lazy_loading
        if not self.lazy_loading:
            self.all_return_dict = [self.get_item(i) for i in range(len(self.indices))]

    def get_item(self, index):
        t1 = default_timer()
        file_index = self.indices[index] if self.indices else str(index).zfill(4)
        return_dict = {}
        file_key_dict = {"pressure": "pressure", "wallshearstress": "wallshearstress"}
        for key in self.data_keys:
            extension = ".pdparams" if key == "info" else ".npy"
            file_key = file_key_dict[key] if key in file_key_dict else key
            return_dict[key] = self.load_file(f"{file_key}_{file_index}", extension)
        try:
            return_dict["df"] = self.load_file(f"df_{file_index}")
            if self.closest_points_to_query:
                return_dict["closest_points"] = self.load_file(f"closest_{file_index}")
        except Exception:
            print(
                "Warning: No 'df' files now, please generate them at first or set 'num_workers=0' for this dataloader."
            )
            return_dict["df"], closest_points = self.get_df_closest(
                self.index_to_mesh_path(file_index)
            )
            if self.closest_points_to_query:
                return_dict["closest_points"] = closest_points
        return_dict["df_query_points"] = paddle.to_tensor(data=self.query_points)
        #if not return_dict["info"]["compute_normal"]:
        return_dict["vertices"] = None
        reference_area = return_dict["info"]["area"]
        areas = self.load_file(f"area_{file_index}")
        centroids = self.load_file(f"centroid_{file_index}")
        triangle_normals = self.load_file(f"normal_{file_index}")
        return_dict["triangle_normals"] = triangle_normals
        mesh_test_path = self.path / f"mesh_rec_{file_index}.ply"
        if get_last_dir(self.path) == "test" and os.path.exists(mesh_test_path):
            mesh = meshio.read(mesh_test_path)
            return_dict["mesh"] = mesh
        # else:
        #     mesh = self.load_mesh(self.index_to_mesh_path(file_index))
        #     vertices = self.vertices_from_mesh(mesh)
        #     triangles = self.triangles_from_mesh(mesh)
        #     centroids, areas = self.get_triangle_centroids(vertices, triangles)
        #     return_dict["vertices"] = vertices
        #     mesh.compute_triangle_normals()
        #     triangle_normals = paddle.to_tensor(
        #         data=mesh.triangle_normals, dtype="float64"
        #     ).reshape((-1, 3))
        #     reference_area = (
        #         return_dict["info"]["width"] * return_dict["info"]["height"] / 2 * 1e-06
        #     )
        flow_directions = paddle.zeros_like(x=triangle_normals)
        flow_directions[:, 0] = -1
        mass_density = return_dict["info"]["density"]
        return_dict["info"]["car_speed"] = float(return_dict["info"]["car_speed"]) / 3.6
        car_speed = float(return_dict["info"]["car_speed"])
        wind_speed = float(return_dict["info"]["wind_speed"])
        wind_angle_rad = math.radians(float(return_dict["info"]["wind_angle"]))
        vx = car_speed + wind_speed * math.cos(wind_angle_rad)
        vy = wind_speed * math.sin(wind_angle_rad)
        flow_speed = math.sqrt(vx**2 + vy**2)
        const = 2.0 / (mass_density * flow_speed**2 * reference_area)
        projection = paddle.sum(
            x=(triangle_normals  * 1e10) * flow_directions, axis=1, keepdim=False
        )
        return_dict["F_const"] = 2.0 / (mass_density * flow_speed**2 * reference_area)
        return_dict["M_const"] = 2.0 / (mass_density * flow_speed**2 * reference_area * float(return_dict["info"]['typical_length']))
        # 参考动压：q_ref = 0.5*rho*flow_speed**2，满足 F_const == 1/(q_ref*reference_area)
        q_ref = compute_q_ref(return_dict["info"])
        return_dict["q_ref"] = q_ref
        return_dict["areas"] = areas
        return_dict["centroids_no_norms"] = centroids
        # 输出物理场(pressure/wallshearstress)先除以 q_ref 转为系数(Cp/Cf)，再 z-score。
        # 其它 key(如 area/location)走纯归一化，不传 q_ref。
        for key in self.norms_dict:
            if key in return_dict:
                if key in self.out_keys:
                    return_dict[key] = self.norms_dict[key](return_dict[key], q_ref=q_ref)
                else:
                    return_dict[key] = self.norms_dict[key](return_dict[key])
        if "location" in self.norms_dict:
            if return_dict["vertices"] is not None:
                return_dict["vertices"] = self.norms_dict["location"](vertices)
            return_dict["centroids"] = self.norms_dict["location"](
                return_dict["centroids_no_norms"]
            )
            return_dict["df_query_points"] = self.norms_dict["location"](
                return_dict["df_query_points"]
            ).transpose(perm=[3, 0, 1, 2])
            if self.closest_points_to_query:
                return_dict["closest_points"] = self.norms_dict["location"](
                    return_dict["closest_points"]
                ).transpose(perm=[3, 0, 1, 2])
        t2 = default_timer()
        return_dict["Data_loading_time"] = t2 - t1
        return return_dict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.lazy_loading:
            return self.get_item(index)
        else:
            return self.all_return_dict[index]


class BaseCFDDataModule(BaseDataModule):

    def __init__(self):
        super().__init__()

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def test_data(self):
        return self._test_data

    def encode(self, norm_fn, data: paddle.Tensor) -> paddle.Tensor:
        norm_fn.to(data.place)
        return norm_fn.encode(data)

    def decode(self, norm_fn, data: paddle.Tensor, q_ref=None) -> paddle.Tensor:
        norm_fn.to(data.place)
        return norm_fn.decode(data, q_ref=q_ref)

    def load_bound(
        self, data_dir, filename="watertight_global_bounds.txt", eps=1e-06
    ) -> Tuple[List[float], List[float]]:
        with open(data_dir / filename, "r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")
            min_bounds = [(float(a) - eps) for a in min_bounds]
            max_bounds = [(float(a) + eps) for a in max_bounds]
        return min_bounds, max_bounds

    def location_normalization(
        self,
        locations: paddle.Tensor,
        min_bounds: Union[paddle.Tensor, List[float]],
        max_bounds: Union[paddle.Tensor, List[float]],
    ) -> paddle.Tensor:
        """
        Normalize locations to [-1, 1].
        """
        if not isinstance(min_bounds, paddle.Tensor):
            min_bounds = paddle.to_tensor(data=min_bounds)
        if not isinstance(max_bounds, paddle.Tensor):
            max_bounds = paddle.to_tensor(data=max_bounds)
        locations = (locations - min_bounds) / (max_bounds - min_bounds)
        locations = 2 * locations - 1
        return locations

    def info_normalization(
        self, info: dict, min_bounds: List[float], max_bounds: List[float]
    ) -> dict:
        """
        Normalize info to [0, 1].
        """
        for i, (k, v) in enumerate(info.items()):
            info[k] = (v - min_bounds[i]) / (max_bounds[i] - min_bounds[i])
        return info

    def area_normalization(
        self, area: paddle.Tensor, min_bounds: float, max_bounds: float
    ) -> paddle.Tensor:
        """
        Normalize area to [0, 1].
        """
        return (area - min_bounds) / (max_bounds - min_bounds)


class SAEDataModule(BaseCFDDataModule):

    def __init__(
        self,
        data_dir,
        out_keys: List[str] = ["pressure"],
        out_channels: List[int] = [1],
        n_train: int = 1,
        n_val: int = 1,
        n_test: int = 1,
        spatial_resolution: Tuple[int, int, int] = None,
        query_points=None,
        closest_points_to_query=False,
        eps=0.01,
        lazy_loading=True,
        train_ratio: float = None,
        test_ratio: float = None,
        train_ids_path: Optional[str] = None,
        test_ids_path: Optional[str] = None,
        split_json_path: Optional[str] = None,
    ):
        super().__init__()
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        data_dir = data_dir.expanduser()
        assert data_dir.exists(), "Path does not exist"
        assert data_dir.is_dir(), "Path is not a directory"
        self.data_dir = data_dir
        self.out_keys = out_keys
        self.out_channels = out_channels
        self.query_points = query_points
        self.closest_points_to_query = closest_points_to_query
        self.spatial_resolution = spatial_resolution
        self.eps = eps
        self.lazy_loading = lazy_loading
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.train_ids_path = Path(train_ids_path) if train_ids_path else None
        self.test_ids_path = Path(test_ids_path) if test_ids_path else None
        self.split_json_path = Path(split_json_path) if split_json_path else None
        self.get_indices(n_train, n_val, n_test)
        self.get_norms(data_dir)
        self.get_data()

    @staticmethod
    def split_list_(input_list, train_ratio, test_ratio):
        # 检查train_ratio和test_ratio之和是否为1
        if not (
            0 <= train_ratio <= 1
            and 0 <= test_ratio <= 1
            and train_ratio + test_ratio == 1.0
        ):
            raise ValueError("train_ratio和test_ratio之和必须为1.0")

        # 随机打乱输入列表
        random.Random(42).shuffle(input_list)

        # 计算训练集的大小
        train_size = int(len(input_list) * train_ratio)

        # 划分列表
        train_list = input_list[:train_size]
        test_list = input_list[train_size:]

        return train_list, test_list

    def load_ids(self, idx_path: str):
        indices = []
        full_caseids = []
        with open(idx_path, "r") as file:
            line = file.readline()
            while line:
                line = line.strip()
                new_index = line.rsplit("-", 1)[0]
                if new_index not in indices:
                    indices.append(new_index)
                full_caseids.append(line)
                line = file.readline()
        return indices, full_caseids

    def init_idx(self, n_data, mode=None, idx_file: Optional[Path] = None) -> List[str]:
        idx_path = idx_file if idx_file is not None else self.data_dir / f"{mode}_design_ids.txt"
        if idx_path.exists():
            indices, full_caseids = self.load_ids(idx_path)
            indices.sort()
            indices = indices[:n_data]
            full_caseids = full_caseids[:n_data]
        else:
            all_files = sorted(os.listdir(self.data_dir))
            all_files = [file for file in all_files if file.endswith(".npy")]
            prefix = "area"
            indices = sorted(set([item[5:-8] for item in all_files if item.startswith(prefix)]))
            indices = indices[:n_data]

            full_caseids = sorted([item[5:-4] for item in all_files if item.startswith(prefix)])
            full_caseids = full_caseids[:n_data]
            # print('indices_%s:' % mode, indices)
        return indices, full_caseids

    def init_data(self, indices, mode="train"):
        data_keys = ["info"]
        data_keys.extend(self.out_keys)
        data_dict = PathDictDataset(
            path=self.data_dir,
            query_points=self.query_points,
            closest_points_to_query=self.closest_points_to_query,
            indices=indices,
            norms_dict=self.norms_dict,
            data_keys=data_keys,
            out_keys=self.out_keys,
            lazy_loading=self.lazy_loading,
        )
        return data_dict

    def get_indices(self, n_train, n_val, n_test):
        if self.split_json_path is not None:
            # 直接复用预训练保存的 radius.json 划分，保证微调与预训练完全一致。
            # radius.json 仅存 design 级 index（train_case_id / test_case_id），
            # full_caseids 需从数据目录重建（与随机划分分支同样的映射: case[:-4] -> index）。
            logging.info(f"Use split from json: {self.split_json_path}.")
            assert self.split_json_path.exists(), f"split_json not found: {self.split_json_path}"
            with open(self.split_json_path, "r") as f:
                split = json.load(f)
            self.train_indices = list(split["train_case_id"])
            self.test_indices = list(split["test_case_id"])

            _, full_caseids = self.init_idx(n_train + n_val + n_test)
            train_set, test_set = set(self.train_indices), set(self.test_indices)
            self.train_full_caseids, self.test_full_caseids = [], []
            for case in full_caseids:
                if case[:-4] in train_set:
                    self.train_full_caseids.append(case)
                elif case[:-4] in test_set:
                    self.test_full_caseids.append(case)
        elif self.train_ids_path is not None and self.test_ids_path is not None:
            logging.info(f"Use the specified dataset: {self.train_ids_path} and {self.test_ids_path}.")
            self.train_indices, self.train_full_caseids = self.init_idx(n_train, "train", self.train_ids_path)
            self.test_indices, self.test_full_caseids = self.init_idx(n_test, "train", self.test_ids_path)
        else:
            logging.info(f"Random generate dataset.")
            fulldata = n_train + n_val + n_test
            full_indices, full_caseids = self.init_idx(fulldata)
            index = list(range(len(full_indices)))
            train_index, test_index = self.split_list_(
                index, train_ratio=self.train_ratio, test_ratio=self.test_ratio
            )
            self.train_indices, self.test_indices = (
                [full_indices[j] for j in train_index],
                [full_indices[k] for k in test_index],
            )

            self.train_full_caseids, self.test_full_caseids = [], []
            for case in full_caseids:
                if case[:-4] in self.train_indices:
                    self.train_full_caseids.append(case)
                else:
                    self.test_full_caseids.append(case)

        print('indices',self.train_indices, self.test_indices)
        print('full_caseids',self.train_full_caseids, self.test_full_caseids)

    def get_norms(self, data_dir):
        min_bounds, max_bounds = self.load_bound(
            data_dir, filename="global_bounds.txt", eps=self.eps
        )
        # min_info_bounds, max_info_bounds = self.load_bound(
        #     data_dir, filename="info_bounds.txt", eps=0.0
        # )
        min_area_bound, max_area_bound = self.load_bound(
            data_dir, filename="area_bounds.txt", eps=0.0
        )
        if self.query_points is None:
            assert (
                self.spatial_resolution is not None
            ), "spatial_resolution must be given"
            tx = np.linspace(min_bounds[0], max_bounds[0], self.spatial_resolution[0])
            ty = np.linspace(min_bounds[1], max_bounds[1], self.spatial_resolution[1])
            tz = np.linspace(min_bounds[2], max_bounds[2], self.spatial_resolution[2])
            self.query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)
        location_norm_fn = lambda x: self.location_normalization(
            x, min_bounds, max_bounds
        )
        info_norm_fn = lambda x: self.info_normalization(
            x, min_info_bounds, max_info_bounds
        )
        area_norm_fn = lambda x: self.area_normalization(
            x, min_area_bound[0], max_area_bound[0]
        )
        self.norms_dict = {"location": location_norm_fn, "area": area_norm_fn}
        self.output_normalization = []
        for i in range(len(self.out_keys)):
            key = self.out_keys[i]
            if key == "pressure":
                file_path = data_dir / f"pressure_{self.train_full_caseids[0]}.npy"
            elif key == "wallshearstress":
                file_path = data_dir / f"wallshearstress_{self.train_full_caseids[0]}.npy"
            # mean/std 描述的是系数(Cp/Cf)的统计量，由预处理写入 *_coef_mean_std.txt
            mean_std_filename = f"train_{key}_coef_mean_std.txt"
            key_normalization = UnitGaussianNormalizer(
                paddle.to_tensor(data=self.load_file(file_path)),
                eps=1e-06,
                reduce_dim=[0],
                verbose=False,
            )
            mean, std = self.load_bound(data_dir, filename=mean_std_filename, eps=0.0)
            key_normalization.mean, key_normalization.std = paddle.to_tensor(
                data=mean[: self.out_channels[i]]
            ), paddle.to_tensor(data=std[: self.out_channels[i]])
            self.norms_dict[key] = copy.deepcopy(key_normalization).encode
            self.output_normalization.append(key_normalization)

    def get_data(self):
        self._train_data = self.init_data(self.train_full_caseids, "train")
        self._test_data = self.init_data(self.test_full_caseids, "test")
        self._aggregatable = ["df", "df_query_points"]

    def load_file(self, file_path: Path) -> np.ndarray:
        assert file_path.exists(), f"File {file_path} does not exist"
        data = np.load(file_path).astype(np.float32)
        return data

    def decode(self, data, idx: int, q_ref=None) -> paddle.Tensor:
        return super().decode(self.output_normalization[idx], data.T, q_ref=q_ref).T

    def collate_fn(self, batch):
        aggr_dict = {}
        for key in self._aggregatable:
            aggr_dict.update(
                {key: paddle.stack(x=[data_dict[key] for data_dict in batch])}
            )
        remaining = list(set(batch[0].keys()) - set(self._aggregatable))
        for key in remaining:
            aggr_dict.update({key: [data_dict[key] for data_dict in batch]})
        return aggr_dict


class TestData(unittest.TestCase):

    def __init__(self, methodName: str, data_path: str) -> None:
        super().__init__(methodName)
        self.data_path = data_path

    def test_ahmed(self):
        dm = SAEDataModule(
            self.data_path, n_train=10, n_test=10, spatial_resolution=(64, 64, 64)
        )
        tl = dm.train_dataloader(batch_size=2, shuffle=True)
        for batch in tl:
            for k, v in batch.items():
                if isinstance(v, paddle.Tensor):
                    print(k, tuple(v.shape))
                else:
                    print(k)
                    for j in range(len(v)):
                        if isinstance(v[j], dict):
                            print(v[j])
                        else:
                            print(tuple(v[j].shape))
            break


if __name__ == "__main__":
    data_dir_test = Path("~/datasets/geono/ahmed").expanduser()
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestData("test_ahmed", data_dir_test))
    unittest.TextTestRunner().run(test_suite)
