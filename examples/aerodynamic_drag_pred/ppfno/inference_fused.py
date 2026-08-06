import gc
import json
import logging
import math
import multiprocessing as mp
import os
import resource
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from timeit import default_timer
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import open3d as o3d
import paddle
import pyvista as pv
from omegaconf import DictConfig, OmegaConf

from ppcfd.models.ppfno.data.datamodule_lazy import compute_q_ref
from ppcfd.models.ppfno.losses import LpLoss
from ppcfd.models.ppfno.networks import instantiate_network
from ppcfd.models.ppfno.neuralop.utils import UnitGaussianNormalizer


CASE_REASON_FILENAME = "reason.json"
DEFAULT_DEBUG_WAIT_TIMEOUT_SECONDS = 60.0


# ======================================================================================
# Case discovery
# ======================================================================================
@dataclass(frozen=True)
class CaseSpec:
    caseid: str
    case_dir: str
    stl_path: str
    json_path: str


def discover_cases(input_root: str) -> List[CaseSpec]:
    """Discover cases under ``input_root``.

    Each case is a sub-directory named ``{caseid}`` containing ``{caseid}.stl``
    and ``{caseid}.json`` (same layout convention as batch_pipeline.py).
    """
    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    cases: List[CaseSpec] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        caseid = entry.name
        stl_path = entry / f"{caseid}.stl"
        json_path = entry / f"{caseid}.json"
        if not stl_path.exists() or not json_path.exists():
            logging.warning(
                "Skipping case %s because required files are missing under %s",
                caseid,
                entry,
            )
            continue
        cases.append(
            CaseSpec(
                caseid=caseid,
                case_dir=str(entry),
                stl_path=str(stl_path),
                json_path=str(json_path),
            )
        )
    return cases


def load_case_info(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


# ======================================================================================
# In-memory preprocessing (raw STL -> area / centroid / normal / distance-field)
#
# Reproduces inferenceDataPreprocessSmesh.py so the fused pipeline consumes raw STL+JSON
# directly instead of pre-generated *.npy files.
# ======================================================================================
def extract_surface_arrays(
    stl_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract per-triangle area / centroid / unit-normal from an STL mesh.

    Matches inferenceDataPreprocessSmesh.py exactly:
      normal = -cross(v1 - v0, v2 - v0) / ||cross||   (inward-pointing convention)
      area   = ||cross(v1 - v0, v2 - v0)|| / 2
      centroid = mean of the three vertices
    """
    mesh_legacy = o3d.io.read_triangle_mesh(stl_path)
    if not mesh_legacy.has_triangles():
        raise RuntimeError(f"Loaded mesh has no triangles: {stl_path}")

    vertices = np.asarray(mesh_legacy.vertices)
    triangles = np.asarray(mesh_legacy.triangles)
    tri_vertices = vertices[triangles]

    v0 = tri_vertices[:, 0]
    v1 = tri_vertices[:, 1]
    v2 = tri_vertices[:, 2]
    centroid = tri_vertices.mean(axis=1).astype(np.float32)

    cross_product = np.cross(v1 - v0, v2 - v0)
    area = (np.linalg.norm(cross_product, axis=1) / 2.0).astype(np.float32)

    norm = np.linalg.norm(cross_product, axis=1, keepdims=True)
    norm = np.clip(norm, a_min=1e-12, a_max=None)
    normal = (-1.0 * cross_product / norm).astype(np.float32)

    surface_mesh_num = int(len(triangles))
    del vertices, triangles, tri_vertices, v0, v1, v2, cross_product, norm
    return area, centroid, normal, surface_mesh_num


def _read_bound(
    bounds_dir: str, filename: str, eps: float
) -> Tuple[List[float], List[float]]:
    with open(os.path.join(bounds_dir, filename), "r", encoding="utf-8") as fp:
        min_bounds = fp.readline().split(" ")
        max_bounds = fp.readline().split(" ")
    min_bounds = [(float(a) - eps) for a in min_bounds]
    max_bounds = [(float(a) + eps) for a in max_bounds]
    return min_bounds, max_bounds


def _build_query_grid(
    bounds_dir: str, spatial_resolution: Tuple[int, int, int], eps: float
) -> np.ndarray:
    min_bounds, max_bounds = _read_bound(bounds_dir, "global_bounds.txt", eps)
    tx = np.linspace(min_bounds[0], max_bounds[0], spatial_resolution[0])
    ty = np.linspace(min_bounds[1], max_bounds[1], spatial_resolution[1])
    tz = np.linspace(min_bounds[2], max_bounds[2], spatial_resolution[2])
    return np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(np.float32)


def compute_distance_field(
    stl_path: str, query_points: np.ndarray
) -> np.ndarray:
    """Unsigned distance field on ``query_points`` via open3d ray-casting.

    Matches Compute_df_stl.compute_df_from_mesh (scene.compute_distance).
    """
    stl_mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh_tensor = o3d.t.geometry.TriangleMesh.from_legacy(stl_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_tensor)
    df = scene.compute_distance(o3d.core.Tensor(query_points)).numpy().astype(np.float32)
    del scene, mesh_tensor, stl_mesh
    return df


def preprocess_case_in_memory(
    case: CaseSpec,
    bounds_dir: str,
    sdf_spatial_resolution: Tuple[int, int, int],
    reason_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    t1 = default_timer()
    debug_case_log(reason_output_path, case.caseid, f"preprocess start, stl={case.stl_path}")

    info = load_case_info(case.json_path)
    debug_case_log(reason_output_path, case.caseid, "json loaded")

    area, centroid, normal, surface_mesh_num = extract_surface_arrays(case.stl_path)
    debug_case_log(
        reason_output_path,
        case.caseid,
        f"surface arrays ready, triangles={surface_mesh_num}",
    )

    # NOTE: distance-field query grid uses eps=1e-6 to match the original
    # Compute_df_stl preprocessing; the model-space normalization grid (eps=0.01)
    # is rebuilt separately in FusedNormalizer.
    df_grid = _build_query_grid(bounds_dir, tuple(sdf_spatial_resolution), eps=1e-6)
    df = compute_distance_field(case.stl_path, df_grid)
    debug_case_log(reason_output_path, case.caseid, f"df ready, shape={tuple(df.shape)}")

    del df_grid
    gc.collect()
    return {
        "caseid": case.caseid,
        "info": info,
        "area": area,
        "centroid": centroid,
        "normal": normal,
        "df": df,
        "surface_mesh_num": surface_mesh_num,
        "preprocess_seconds": default_timer() - t1,
    }


# ======================================================================================
# SAE-format model-input builder / decoder
#
# Reproduces SAEInferenceDataModule.get_norms + PathDictDataset.get_item so the ppcfd
# GNOFNOGNO_all network receives exactly the data_dict layout it was trained with.
# ======================================================================================
class FusedNormalizer:
    def __init__(
        self,
        bounds_dir: str,
        out_keys: List[str],
        out_channels: List[int],
        spatial_resolution: Tuple[int, int, int],
        eps: float = 0.01,
    ):
        self.bounds_dir = bounds_dir
        self.out_keys = out_keys
        self.out_channels = out_channels
        self.spatial_resolution = spatial_resolution
        self.eps = eps

        self.min_bounds, self.max_bounds = _read_bound(
            bounds_dir, "global_bounds.txt", eps=eps
        )
        # model-space query grid (normalized to [-1, 1] below)
        self.query_points = _build_query_grid(
            bounds_dir, tuple(spatial_resolution), eps=eps
        )
        self.output_normalization = self._build_output_normalizers()

    def _build_output_normalizers(self) -> List[UnitGaussianNormalizer]:
        normalizers: List[UnitGaussianNormalizer] = []
        for i, key in enumerate(self.out_keys):
            if key == "pressure":
                data = np.zeros((100,), dtype=np.float32)
            elif key == "wallshearstress":
                data = np.zeros((100, 3), dtype=np.float32)
            else:
                raise ValueError(f"Unsupported output key: {key}")
            norm = UnitGaussianNormalizer(
                paddle.to_tensor(data=data), eps=1e-6, reduce_dim=[0], verbose=False
            )
            # mean/std describe coefficient (Cp/Cf) statistics, written by preprocessing.
            mean, std = _read_bound(
                self.bounds_dir, f"train_{key}_coef_mean_std.txt", eps=0.0
            )
            norm.mean = paddle.to_tensor(data=mean[: self.out_channels[i]], dtype="float32")
            norm.std = paddle.to_tensor(data=std[: self.out_channels[i]], dtype="float32")
            normalizers.append(norm)
        return normalizers

    def location_normalization(self, locations: paddle.Tensor) -> paddle.Tensor:
        min_bounds = paddle.to_tensor(data=self.min_bounds, dtype="float32")
        max_bounds = paddle.to_tensor(data=self.max_bounds, dtype="float32")
        locations = (locations - min_bounds) / (max_bounds - min_bounds)
        return 2 * locations - 1

    def decode(self, data: paddle.Tensor, idx: int, q_ref=None) -> paddle.Tensor:
        norm_fn = self.output_normalization[idx]
        norm_fn.to(data.place)
        return norm_fn.decode(data.T, q_ref=q_ref).T

    def build_model_input(
        self, caseid: str, preprocessed: Dict[str, Any]
    ) -> Dict[str, Any]:
        info = dict(preprocessed["info"])
        # car_speed is stored in km/h in the JSON; convert to m/s once (matches datamodule).
        info["car_speed"] = float(info["car_speed"]) / 3.6

        centroids_raw = paddle.to_tensor(preprocessed["centroid"].astype(np.float32))
        areas = paddle.to_tensor(preprocessed["area"].astype(np.float32))
        normals = paddle.to_tensor(preprocessed["normal"].astype(np.float32))
        df = paddle.to_tensor(preprocessed["df"].astype(np.float32))

        centroids_norm = self.location_normalization(centroids_raw)
        query_points = paddle.to_tensor(self.query_points.copy())
        df_query_points = self.location_normalization(query_points).transpose(
            perm=[3, 0, 1, 2]
        )

        return {
            "info": [info],
            "df": paddle.stack(x=[df]),                     # [1, Nx, Ny, Nz]
            "df_query_points": paddle.stack(x=[df_query_points]),  # [1, 3, Nx, Ny, Nz]
            "vertices": [None],
            "areas": [areas],
            "centroids": [centroids_norm],
            "centroids_no_norms": [centroids_raw],
            "triangle_normals": [normals],
            "caseid": [caseid],
        }


# ======================================================================================
# Output directory / logging helpers (per-case isolation, mirrors batch_pipeline.py)
# ======================================================================================
def get_case_output_dir(reason_output_path: str, caseid: str) -> str:
    return os.path.join(reason_output_path, caseid)


def get_case_json_dir(reason_output_path: str, caseid: str) -> str:
    return os.path.join(get_case_output_dir(reason_output_path, caseid), "json")


def get_case_vtp_csv_dir(reason_output_path: str, caseid: str) -> str:
    return os.path.join(get_case_output_dir(reason_output_path, caseid), "csv_vtp")


def get_case_log_dir(reason_output_path: str, caseid: str) -> str:
    return os.path.join(get_case_output_dir(reason_output_path, caseid), "log")


def get_case_reason_path(reason_output_path: str, caseid: str, filename: str) -> str:
    return os.path.join(get_case_json_dir(reason_output_path, caseid), filename)


def get_case_log_path(reason_output_path: str, caseid: str) -> str:
    return os.path.join(get_case_log_dir(reason_output_path, caseid), f"{caseid}.txt")


def ensure_case_output_dirs(reason_output_path: str, caseid: str) -> Dict[str, str]:
    case_dir = get_case_output_dir(reason_output_path, caseid)
    json_dir = get_case_json_dir(reason_output_path, caseid)
    vtp_csv_dir = get_case_vtp_csv_dir(reason_output_path, caseid)
    log_dir = get_case_log_dir(reason_output_path, caseid)
    for path in (case_dir, json_dir, vtp_csv_dir, log_dir):
        os.makedirs(path, exist_ok=True)
    return {
        "case_dir": case_dir,
        "json_dir": json_dir,
        "vtp_csv_dir": vtp_csv_dir,
        "log_dir": log_dir,
    }


def get_memory_gb() -> float:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss_kb / 1024 / 1024


def append_debug_log(log_path: str, level: str, message: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    with open(log_path, "a", encoding="utf-8") as fp:
        fp.write(f"[{timestamp}][{level}][rss={get_memory_gb():.2f}GB] - {message}\n")
        fp.flush()


def debug_case_log(reason_output_path: Optional[str], caseid: str, message: str) -> None:
    if not reason_output_path:
        return
    append_debug_log(
        get_case_log_path(reason_output_path, caseid),
        "INFO",
        f"[{caseid}][pid={os.getpid()}] {message}",
    )


def initialize_case_log(reason_output_path: str, caseid: str, cfg: DictConfig) -> None:
    log_path = get_case_log_path(reason_output_path, caseid)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write("================= FUSED SAE INFERENCE PIPELINE =================\n")
        fp.write(f"caseid: {caseid}\n")
        fp.write("config:\n")
        fp.write(OmegaConf.to_yaml(cfg))
        if not str(OmegaConf.to_yaml(cfg)).endswith("\n"):
            fp.write("\n")
        fp.write("===============================================================\n")
        fp.flush()


# ======================================================================================
# Result saving (decoded pressure / wallshearstress fields -> csv + vtp)
# ======================================================================================
def save_vtp_from_dict(
    filename: str,
    data_dict: Dict[str, np.ndarray],
    coord_keys: Tuple[str, ...],
    value_keys: Tuple[str, ...],
) -> None:
    coord = [data_dict[k] for k in coord_keys]
    coord = np.concatenate(coord, axis=1)
    if len(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if filename.endswith(".vtp"):
        filename = filename[:-4]
    point_cloud = pv.PolyData(coord)
    for k in value_keys:
        point_cloud[k] = data_dict[k]
    point_cloud.save(f"{filename}.vtp", binary=True)


def save_eval_results(
    reason_output_path: str,
    caseid: str,
    value: float,
    centroid_raw: np.ndarray,
    pred: paddle.Tensor,
    decode_fn,
    q_ref,
    subsample_eval_out: int = 1,
) -> None:
    pred_pressure = decode_fn(pred[0:1, :], 0, q_ref=q_ref).cpu().detach().numpy()
    pred_wallshearstress = decode_fn(pred[1:4, :], 1, q_ref=q_ref).cpu().detach().numpy()
    evals_results = {
        "cal_pressure_drag": pred_pressure,
        "cal_friction_resistance": pred_wallshearstress,
    }

    centroid = centroid_raw[:: subsample_eval_out, ...]
    out_dir = os.path.join(
        get_case_vtp_csv_dir(reason_output_path, caseid), str(int(value))
    )
    os.makedirs(out_dir, exist_ok=True)

    for k, v in evals_results.items():
        array_hstack = np.hstack((centroid, v.T))
        csv_filename = os.path.join(out_dir, f"{k}.csv")
        np.savetxt(csv_filename, array_hstack, delimiter=",", fmt="%f")
        vtp_filename = os.path.join(out_dir, f"{k}.vtp")
        if v.T.shape[1] == 1:
            field = v.T
        else:
            field = np.linalg.norm(v.T, axis=1, keepdims=True)
        save_vtp_from_dict(
            vtp_filename,
            {
                "x": centroid[:, 0:1],
                "y": centroid[:, 1:2],
                "z": centroid[:, 2:3],
                k: field,
            },
            ("x", "y", "z"),
            (k,),
        )


# ======================================================================================
# Per-case inference (with wind_speed / wind_angle scanning + full F/M outputs)
# ======================================================================================
def _build_load_entry(cal_val, cal_val_modify, const):
    """真·系数空间：modify 分支输出无量纲系数，×(1/const) 还原物理值；
    no_modify 分支是骨干积分的物理力，×const 得系数。"""
    return {
        "cal_value": float(cal_val_modify) / const,
        "coefficient": float(cal_val_modify),
        "cal_value_no_modify": float(cal_val),
        "coefficient_no_modify": float(cal_val) * const,
    }


def run_case_inference(
    cfg: DictConfig,
    model: paddle.nn.Layer,
    loss_fn: LpLoss,
    normalizer: FusedNormalizer,
    case: CaseSpec,
    preprocessed: Dict[str, Any],
) -> Dict[str, Any]:
    case_dirs = ensure_case_output_dirs(cfg.reason_output_path, case.caseid)
    case_output_dir = case_dirs["case_dir"]
    case_t0 = default_timer()

    debug_case_log(cfg.reason_output_path, case.caseid, "start building model input")
    data_dict = normalizer.build_model_input(case.caseid, preprocessed)
    debug_case_log(cfg.reason_output_path, case.caseid, "model input ready")

    device = paddle.CUDAPlace(0) if paddle.device.is_compiled_with_cuda() else paddle.CPUPlace()

    # Determine the scanning dimension (wind_speed or wind_angle) from comma lists.
    if "," in str(data_dict["info"][0]["wind_speed"]):
        value_list = [float(v) for v in str(data_dict["info"][0]["wind_speed"]).split(",")]
        value_type = "wind_speed"
    else:
        value_list = [float(v) for v in str(data_dict["info"][0]["wind_angle"]).split(",")]
        value_type = "wind_angle"

    inference_records: List[Dict[str, Any]] = []
    t_infer_total = 0.0
    for value in value_list:
        record: Dict[str, Any] = {"type": value_type}
        data_dict["info"][0][value_type] = value
        # 扫描值写回 info 后重算 q_ref，保证 decode 用的 q_ref 与积分用的 F_const 一致。
        data_dict["q_ref"] = [compute_q_ref(data_dict["info"][0])]

        t1 = default_timer()
        out_dict, pred, F_M_dict = model.inference_dict(
            device, data_dict, loss_fn=loss_fn, decode_fn=normalizer.decode
        )
        t_infer_total += default_timer() - t1

        if cfg.save_eval_results:
            save_eval_results(
                cfg.reason_output_path,
                case.caseid,
                value,
                preprocessed["centroid"],
                pred,
                decode_fn=normalizer.decode,
                q_ref=data_dict["q_ref"][0],
                subsample_eval_out=int(cfg.get("subsample_eval_out", 1)),
            )

        # 折算物理常数（与网络内部 inference_dict 保持一致）
        mass_density = float(data_dict["info"][0]["density"])
        reference_area = float(data_dict["info"][0]["area"])
        typical_length = float(data_dict["info"][0]["typical_length"])
        car_speed = float(data_dict["info"][0]["car_speed"])  # already m/s
        wind_speed = float(data_dict["info"][0]["wind_speed"])
        wind_angle_rad = math.radians(float(data_dict["info"][0]["wind_angle"]))
        vx = car_speed + wind_speed * math.cos(wind_angle_rad)
        vy = wind_speed * math.sin(wind_angle_rad)
        flow_speed = math.sqrt(vx ** 2 + vy ** 2)
        F_const = 2.0 / (mass_density * flow_speed ** 2 * reference_area)
        M_const = 2.0 / (mass_density * flow_speed ** 2 * reference_area * typical_length)

        record["car_speed"] = data_dict["info"][0]["car_speed"]
        record["wind_speed"] = data_dict["info"][0]["wind_speed"]
        record["wind_angle"] = data_dict["info"][0]["wind_angle"]

        def to_np(x):
            return x.numpy() if hasattr(x, "numpy") else x

        load_types = {
            "aerodynamic_lift": (F_M_dict["F_pred"][1], F_M_dict["F_pred_modify"][1], F_const),
            "aerodynamic_drag": (F_M_dict["F_pred"][0], F_M_dict["F_pred_modify"][0], F_const),
            "pneumatic_lateral_force": (F_M_dict["F_pred"][2], F_M_dict["F_pred_modify"][2], F_const),
            "pneumatic_overturning_moment": (F_M_dict["M_pred"][0], F_M_dict["M_pred_modify"][0], M_const),
            "pneumatic_pitching_moment": (F_M_dict["M_pred"][1], F_M_dict["M_pred_modify"][1], M_const),
            "pneumatic_roll_moment": (F_M_dict["M_pred"][2], F_M_dict["M_pred_modify"][2], M_const),
        }
        for load_name, (raw_val, modify_val, const) in load_types.items():
            record[load_name] = _build_load_entry(to_np(raw_val), to_np(modify_val), const)

        inference_records.append(record)
        debug_case_log(
            cfg.reason_output_path,
            case.caseid,
            f"scan {value_type}={value} done, drag_coef={record['aerodynamic_drag']['coefficient']:.4f}",
        )

    total_seconds = default_timer() - case_t0
    reason = {
        "caseid": case.caseid,
        "status": "success",
        "case_output_dir": case_output_dir,
        "preprocess_seconds": preprocessed.get("preprocess_seconds"),
        "inference_seconds": t_infer_total,
        "total_seconds": total_seconds,
        "surface_mesh_num": preprocessed.get("surface_mesh_num"),
        "scan_type": value_type,
        "results": inference_records,
    }
    write_json(get_case_reason_path(cfg.reason_output_path, case.caseid, CASE_REASON_FILENAME), reason)
    debug_case_log(cfg.reason_output_path, case.caseid, "case reason.json saved")

    cleanup_case_memory(data_dict)
    return reason


def create_failure_reason(
    reason_output_path: str,
    caseid: str,
    error: str,
    preprocess_seconds: Optional[float] = None,
    surface_mesh_num: Optional[int] = None,
) -> Dict[str, Any]:
    ensure_case_output_dirs(reason_output_path, caseid)
    reason = {
        "caseid": caseid,
        "status": "failed",
        "case_output_dir": get_case_output_dir(reason_output_path, caseid),
        "preprocess_seconds": preprocess_seconds,
        "surface_mesh_num": surface_mesh_num,
        "error": error,
    }
    write_json(get_case_reason_path(reason_output_path, caseid, CASE_REASON_FILENAME), reason)
    return reason


# ======================================================================================
# JSON / memory utilities
# ======================================================================================
def tensor_to_scalar(value: Any) -> Any:
    if isinstance(value, paddle.Tensor):
        if tuple(value.shape) == () or value.numel() == 1:
            return value.item()
        return value.cpu().numpy().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return tensor_to_scalar(value)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(make_json_safe(data), fp, ensure_ascii=False, indent=2)


def cleanup_case_memory(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if paddle.device.is_compiled_with_cuda() and hasattr(paddle.device.cuda, "empty_cache"):
        paddle.device.cuda.empty_cache()


# ======================================================================================
# Pipelines (streaming + parallel preprocessing)
# ======================================================================================
def run_streaming_pipeline(
    cfg: DictConfig,
    model: paddle.nn.Layer,
    loss_fn: LpLoss,
    normalizer: FusedNormalizer,
    cases: List[CaseSpec],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for case in cases:
        ensure_case_output_dirs(cfg.reason_output_path, case.caseid)
        preprocessed = None
        try:
            logging.info("[%s] Start preprocessing", case.caseid)
            preprocessed = preprocess_case_in_memory(
                case,
                cfg.bounds_dir,
                tuple(cfg.sdf_spatial_resolution),
                reason_output_path=cfg.reason_output_path,
            )
            logging.info("[%s] Start inference", case.caseid)
            reason = run_case_inference(cfg, model, loss_fn, normalizer, case, preprocessed)
            results.append(reason)
            logging.info("[%s] Finished successfully", case.caseid)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logging.exception("[%s] Failed during streaming pipeline", case.caseid)
            results.append(
                create_failure_reason(
                    cfg.reason_output_path,
                    case.caseid,
                    error,
                    preprocess_seconds=preprocessed.get("preprocess_seconds") if preprocessed else None,
                    surface_mesh_num=preprocessed.get("surface_mesh_num") if preprocessed else None,
                )
            )
        finally:
            cleanup_case_memory(preprocessed)
    return results


def run_parallel_preprocess_pipeline(
    cfg: DictConfig,
    model: paddle.nn.Layer,
    loss_fn: LpLoss,
    normalizer: FusedNormalizer,
    cases: List[CaseSpec],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    max_workers = int(cfg.preprocess_workers)
    in_flight: Dict[Any, Dict[str, Any]] = {}
    remaining = iter(cases)
    wait_timeout = DEFAULT_DEBUG_WAIT_TIMEOUT_SECONDS

    spawn_context = mp.get_context("spawn")
    logging.info("Parallel preprocess workers=%d (spawn)", max_workers)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_context) as executor:
        while True:
            while len(in_flight) < max_workers:
                try:
                    case = next(remaining)
                except StopIteration:
                    break
                ensure_case_output_dirs(cfg.reason_output_path, case.caseid)
                future = executor.submit(
                    preprocess_case_in_memory,
                    case,
                    cfg.bounds_dir,
                    tuple(cfg.sdf_spatial_resolution),
                    cfg.reason_output_path,
                )
                in_flight[future] = {"case": case, "submitted_at": default_timer()}
                logging.info("[%s] Submitted preprocess task", case.caseid)

            if not in_flight:
                break

            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED, timeout=wait_timeout)
            if not done:
                logging.info("No preprocess task completed in %.1fs. in_flight=%d", wait_timeout, len(in_flight))
                continue

            for future in done:
                meta = in_flight.pop(future)
                case = meta["case"]
                preprocessed = None
                try:
                    preprocessed = future.result()
                    logging.info("[%s] Preprocess done; start inference", case.caseid)
                    reason = run_case_inference(cfg, model, loss_fn, normalizer, case, preprocessed)
                    results.append(reason)
                    logging.info("[%s] Finished successfully", case.caseid)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logging.exception("[%s] Failed during parallel pipeline", case.caseid)
                    results.append(
                        create_failure_reason(
                            cfg.reason_output_path,
                            case.caseid,
                            error,
                            preprocess_seconds=preprocessed.get("preprocess_seconds") if preprocessed else None,
                            surface_mesh_num=preprocessed.get("surface_mesh_num") if preprocessed else None,
                        )
                    )
                finally:
                    cleanup_case_memory(preprocessed)
    return results


# ======================================================================================
# Entry point
# ======================================================================================
def set_seed(seed: int = 0) -> None:
    paddle.seed(seed=seed)
    np.random.seed(seed)


def quote_non_ascii_overrides(argv: List[str]) -> List[str]:
    """给含非 ASCII 字符（如中文路径）的 Hydra 覆盖参数值自动加引号。"""
    result = []
    for arg in argv:
        if arg.startswith("-") or "=" not in arg:
            result.append(arg)
            continue
        key, sep, value = arg.partition("=")
        already_quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in "'\""
        if value and not already_quoted and not value.isascii():
            value = f'"{value}"'
        result.append(key + sep + value)
    return result


@hydra.main(version_base=None, config_path="./configs", config_name="inference_fused")
def main(cfg: DictConfig) -> None:
    os.makedirs(cfg.reason_output_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(filename)s][%(lineno)d] - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info(" ============== FUSED SAE INFERENCE PIPELINE ============== ")
    logging.info("Fused pipeline config:\n%s", OmegaConf.to_yaml(cfg))

    if cfg.enable_ddp:
        logging.warning("enable_ddp=true is ignored in fused pipeline; forcing single-process inference.")
    if cfg.seed is not None:
        set_seed(cfg.seed)

    assert cfg.state is not None, "checkpoint (cfg.state) must be given."
    cases = discover_cases(cfg.pre_input_path)
    if not cases:
        raise RuntimeError(f"No valid cases found under {cfg.pre_input_path}")
    logging.info("Discovered %d cases under %s", len(cases), cfg.pre_input_path)
    for case in cases:
        ensure_case_output_dirs(cfg.reason_output_path, case.caseid)
        initialize_case_log(cfg.reason_output_path, case.caseid, cfg)

    normalizer = FusedNormalizer(
        bounds_dir=cfg.bounds_dir,
        out_keys=list(cfg.out_keys),
        out_channels=list(cfg.out_channels),
        spatial_resolution=tuple(cfg.sdf_spatial_resolution),
        eps=0.01,
    )

    logging.info("Loading checkpoint from: %s", cfg.state)
    model = instantiate_network(cfg)
    loss_fn = LpLoss(size_average=True)
    state = paddle.load(path=str(cfg.state))
    model.set_state_dict(state_dict=state["model"])
    if isinstance(model, paddle.DataParallel):
        model = model._layers
    model.eval()

    if paddle.device.is_compiled_with_cuda():
        mem = paddle.device.cuda.memory_allocated() / (1024 ** 3)
        logging.info("Memory usage with model loading: %.2f GB", mem)

    try:
        with paddle.no_grad():
            if cfg.preprocess_workers > 1:
                summary = run_parallel_preprocess_pipeline(cfg, model, loss_fn, normalizer, cases)
            else:
                summary = run_streaming_pipeline(cfg, model, loss_fn, normalizer, cases)
    except Exception as exc:
        logging.error("Pipeline failed with unexpected error: %s", exc)
        logging.error(traceback.format_exc())
        raise

    success = sum(1 for item in summary if item.get("status") == "success")
    failed = len(summary) - success
    logging.info("Pipeline finished. success=%d failed=%d total=%d", success, failed, len(summary))


if __name__ == "__main__":
    sys.argv = quote_non_ascii_overrides(sys.argv)
    main()
