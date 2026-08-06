import os
import sys
import paddle
import logging
import random
import pandas as pd
import numpy as np
import re
import multiprocessing as mp
from multiprocessing import Pool, Manager, cpu_count, get_context
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
 
import meshio
import open3d as o3d
import hydra
from omegaconf import DictConfig,OmegaConf,ListConfig
import json
from timeit import default_timer
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import math
import traceback
 
 
class PreprocessProgress:
    """
    Track and save preprocessing progress to JSON file.
 
    Attributes:
        progress_file: Path to save progress JSON
        data: Dictionary containing all progress information
    """
 
    def __init__(self, output_path: str):
        """
        Initialize progress tracker.
 
        Args:
            output_path: Base output directory for preprocessing
        """
        # Ensure log directory exists
        log_dir = os.path.join(output_path, "log")
        os.makedirs(log_dir, exist_ok=True)
 
        self.progress_file = os.path.join(log_dir, "progress.json")
 
        # Initialize progress data structure
        self.data = {
            "metadata": {
                "preprocess_start_time": None,
                "preprocess_end_time": None,
                "config": {}
            },
            "statistics": {
                "total_cases": 0,
                "valid_cases": 0,
                "skipped_cases": 0,
                "case_distribution": {},
                "skipped_case_ids": []
            },
            "progress": {
                "current_stage": "bounds_computation",
                "overall_percentage": 0.0,
                "stage_weights": {
                    "bounds_computation": 50,
                    "feature_extraction": 50
                },
                "stages": {
                    "bounds_computation": {
                        "status": "pending",
                        "start_time": None,
                        "end_time": None,
                        "total_cases": 0,
                        "processed_cases": 0,
                        "valid_cases": 0,
                        "skipped_cases": 0,
                        "skipped_case_ids": [],
                        "progress_percentage": 0.0
                    },
                    "feature_extraction": {
                        "status": "pending",
                        "start_time": None,
                        "end_time": None,
                        "current_case": "",
                        "total_cases": 0,
                        "processed_cases": 0,
                        "progress_percentage": 0.0
                    }
                }
            },
            "output_files": {
                "case_statistics": "",
                "drag_coefficients_json": "",
                "drag_coefficients_md": "",
                "bounds_files": [],
                "progress_file": self.progress_file
            },
            "performance": {
                "total_elapsed_time_seconds": 0.0,
                "average_time_per_case_seconds": 0.0,
                "stage_times": {}
            }
        }
 
        # Track start time
        self.start_time = time.time()
        self.data["metadata"]["preprocess_start_time"] = self._get_timestamp()
 
        # Save initial progress file
        self._save()
 
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
 
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update configuration metadata.
 
        Args:
            config: Configuration dictionary
        """
        self.data["metadata"]["config"] = config
        self._save()
 
    def start_stage(self, stage_name: str, details: str = "", total_cases: int = 0) -> None:
        """
        Mark the start of a processing stage.
 
        Args:
            stage_name: Name of the stage
            details: Optional details about the stage
            total_cases: Total number of cases to process in this stage
        """
        if stage_name in self.data["progress"]["stages"]:
            self.data["progress"]["current_stage"] = stage_name
            stage = self.data["progress"]["stages"][stage_name]
            stage["status"] = "in_progress"
            stage["start_time"] = self._get_timestamp()
            stage["details"] = details
 
            # Set total cases for the stage
            if total_cases > 0:
                stage["total_cases"] = total_cases
                stage["processed_cases"] = 0
 
            self._save()
 
    def end_stage(self, stage_name: str, details: str = "") -> None:
        """
        Mark the end of a processing stage.
 
        Args:
            stage_name: Name of the stage
            details: Optional completion details
        """
        if stage_name in self.data["progress"]["stages"]:
            stage = self.data["progress"]["stages"][stage_name]
            stage["status"] = "completed"
            stage["end_time"] = self._get_timestamp()
 
            # Mark stage as fully processed
            if "total_cases" in stage and stage["total_cases"] > 0:
                stage["processed_cases"] = stage["total_cases"]
 
            if details:
                stage["details"] = details
 
            self._save()
 
    def update_case_statistics(self, case_distribution: Dict[str, int], total_cases: int) -> None:
        """
        Update case statistics information.
 
        Args:
            case_distribution: Dictionary of case categories and counts
            total_cases: Total number of cases
        """
        self.data["statistics"]["total_cases"] = total_cases
        self.data["statistics"]["case_distribution"] = case_distribution
        self._save()
 
    def update_bounds_computation_progress(self, processed_cases: int, total_cases: int) -> None:
        """
        Update bounds computation progress.
 
        Args:
            processed_cases: Number of cases already processed for statistics aggregation.
            total_cases: Total number of cases participating in bounds computation.
        """
        stage = self.data["progress"]["stages"]["bounds_computation"]
        stage["processed_cases"] = processed_cases
        stage["total_cases"] = total_cases
        self._save()
 
    def update_bounds_computation_results(self, valid_cases: int, skipped_cases: int,
                                         skipped_case_ids: List[str]) -> None:
        """
        Update bounds computation results.
 
        Args:
            valid_cases: Number of cases loaded successfully for aggregation.
            skipped_cases: Number of cases skipped during aggregation.
            skipped_case_ids: List of skipped case IDs.
        """
        self.data["statistics"]["valid_cases"] = valid_cases
        self.data["statistics"]["skipped_cases"] = skipped_cases
        self.data["statistics"]["skipped_case_ids"] = skipped_case_ids
 
        stage = self.data["progress"]["stages"]["bounds_computation"]
        stage["valid_cases"] = valid_cases
        stage["skipped_cases"] = skipped_cases
        stage["skipped_case_ids"] = skipped_case_ids
        self._save()
 
    def update_feature_extraction_progress(self, current_case: str, processed: int,
                                          total: int) -> None:
        """
        Update feature extraction progress.
 
        Args:
            current_case: Current case being processed
            processed: Number of cases processed
            total: Total number of cases to process
        """
        stage = self.data["progress"]["stages"]["feature_extraction"]
        stage["current_case"] = current_case
        stage["processed_cases"] = processed
        stage["total_cases"] = total
        self._save()
 
    def update_output_files(self, file_type: str, file_path: str) -> None:
        """
        Update output file information.
 
        Args:
            file_type: Type of file (case_statistics, drag_coefficients_json, etc.)
            file_path: Path to the file
        """
        if file_type in self.data["output_files"]:
            if isinstance(self.data["output_files"][file_type], list):
                self.data["output_files"][file_type].append(file_path)
            else:
                self.data["output_files"][file_type] = file_path
        self._save()
 
    def complete(self) -> None:
        """Mark preprocessing as completed."""
        self.data["metadata"]["preprocess_end_time"] = self._get_timestamp()
 
        # Calculate performance metrics
        total_time = time.time() - self.start_time
        self.data["performance"]["total_elapsed_time_seconds"] = round(total_time, 2)
 
        valid_cases = self.data["statistics"]["valid_cases"]
        if valid_cases > 0:
            avg_time = total_time / valid_cases
            self.data["performance"]["average_time_per_case_seconds"] = round(avg_time, 2)
 
        # Calculate stage times
        for stage_name, stage_info in self.data["progress"]["stages"].items():
            if stage_info["start_time"] and stage_info["end_time"]:
                start = datetime.fromisoformat(stage_info["start_time"])
                end = datetime.fromisoformat(stage_info["end_time"])
                duration = (end - start).total_seconds()
                self.data["performance"]["stage_times"][stage_name] = round(duration, 2)
 
        self._save()
 
    def _calculate_stage_progress(self, stage_name: str) -> float:
        """
        Calculate progress percentage for a specific stage (0-100).
 
        Args:
            stage_name: Name of the stage
 
        Returns:
            Progress percentage for the stage
        """
        stage = self.data["progress"]["stages"][stage_name]
 
        if stage["status"] == "completed":
            return 100.0
        elif stage["status"] == "in_progress":
            total = stage.get("total_cases", 0)
            processed = stage.get("processed_cases", 0)
 
            if total > 0:
                return (processed / total) * 100.0
            else:
                return 0.0
        else:
            # pending status
            return 0.0
 
    def _calculate_overall_progress(self) -> float:
        """
        Calculate overall progress percentage (0-100).
 
        Returns:
            Overall progress percentage
        """
        overall = 0.0
        stage_weights = self.data["progress"]["stage_weights"]
 
        for stage_name, weight in stage_weights.items():
            stage_progress = self._calculate_stage_progress(stage_name)
            overall += (weight * stage_progress / 100.0)
 
        return math.floor(overall)
 
    def _update_progress_percentages(self) -> None:
        """Update progress percentages for all stages and overall."""
        # Update stage progress percentages
        for stage_name in self.data["progress"]["stages"]:
            stage_progress = self._calculate_stage_progress(stage_name)
            self.data["progress"]["stages"][stage_name]["progress_percentage"] = stage_progress
 
        # Update overall progress percentage
        self.data["progress"]["overall_percentage"] = self._calculate_overall_progress()
 
    def _save(self) -> None:
        """Save progress data to JSON file."""
        try:
            # Update progress percentages before saving
            self._update_progress_percentages()
 
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # Log error but don't crash the main process
            logging.getLogger().warning(f"Failed to save progress file: {e}")
 
 
class CFDDataTransiton:
    def __init__(self,case_path,save_path,caseID):
        self.inward_surface_normal = None
        self.cell_area = None
        self.filename_prefix = caseID
        self.case_path = case_path
        self.save_path = save_path
        # Construct full filename prefix from parent directory name and caseID
        # case_path is like: /path/to/SFE-CR400AF_S-U3-FZ-F087/001
        # parent_dir is: SFE-CR400AF_S-U3-FZ-F087
        # filename_prefix should be: SFE-CR400AF_S-U3-FZ-F087-001
        self.filename_prefix = get_case_filename_prefix(case_path, caseID)
        if not self.get_csv_data():
            raise ValueError("CSV data load failed.")
        if not self.get_info():
            raise ValueError("JSON info load failed.")
        self.centroid = self.csv_data[:, -3:]
        self.press = self.csv_data[:, 0]
        self.wallshearstress = self.csv_data[:, 1:4] * -1
        self.cell_area_ijk = self.csv_data[:, 4:7]
        self.flow_direction = np.array([-1, 0, 0])
        self.lift_direction = np.array([0, 0, 1])
 
        if "car_speed" in self.info and "wind_speed" in self.info:
            self.velocity = math.sqrt(self.info["car_speed"]**2 + self.info["wind_speed"]**2)
        else:
            self.velocity = self.info["velocity"]
 
        self.reference_area = self.info.get("area", self.info.get("reference_area"))
        if self.reference_area is None:
            raise ValueError("reference area is missing from json info")
        self.density = self.info["density"]
        self.const = 2.0 / (
            self.density * self.velocity**2 * self.reference_area
        )
        self.drag_c()
        self.lift_c()
 
    def get_csv_data(self):
        csv_file = os.path.join(self.case_path, f'{self.filename_prefix}.csv')
        if not os.path.exists(csv_file):
            logging.error(f"{self.filename_prefix+'.csv'} exist.")
            return False
        try:
            converters = {col:lambda x:float(x) for col in pd.read_csv(csv_file,nrows=0).columns}
        except:
            logging.error(f"{self.filename_prefix+'.csv'} maybe empty.")
            return False
        df = pd.read_csv(csv_file,converters=converters,header=0)
        col_set1 = [
                    "Pressure (Pa)",
                    "Wall Shear Stress[i] (Pa)",
                    "Wall Shear Stress[j] (Pa)",
                    "Wall Shear Stress[k] (Pa)",
                    "Area[i] (m^2)",
                    "Area[j] (m^2)",
                    "Area[k] (m^2)",
                    "X (m)",
                    "Y (m)",
                    "Z (m)"
                    ]
        col_set2 = [
                    "Mean of Pressure (Pa)",
                    "Mean of Wall Shear Stress[i] (Pa)",
                    "Mean of Wall Shear Stress[j] (Pa)",
                    "Mean of Wall Shear Stress[k] (Pa)",
                    "Area[i] (m^2)",
                    "Area[j] (m^2)",
                    "Area[k] (m^2)",
                    "X (m)",
                    "Y (m)",
                    "Z (m)"
                    ]
        if all(col in df.columns for col in col_set1):
            self.csv_data =  df[col_set1].to_numpy()
            return True
        elif all(col in df.columns for col in col_set2):
            self.csv_data = df[col_set2].to_numpy()
            return True
        else:
            logging.error(f"{self.filename_prefix+'.csv'} may not contain the desired data.")
            return False
 
    def get_info(self):
        json_file_path = os.path.join(self.case_path, self.filename_prefix + ".json")
        if not os.path.exists(json_file_path):
            logging.error(f"{self.filename_prefix+'.json'} not exist.")
            return False
        try:
            with open(json_file_path, "r", encoding="utf-8") as file:
                self.info = json.load(file)
            return True
        except:
            logging.error(f"{self.filename_prefix+'.json'} maybe empty.")
            return False
 
    @property
    def area(self):
        # self.cell_area = np.sqrt(np.sum(self.cell_area_ijk**2,axis=1))
        self.cell_area = np.linalg.norm(self.cell_area_ijk,axis=1)
        return self.cell_area
 
    @property
    def normal(self):
        self.inward_surface_normal = (
            -1 * self.cell_area_ijk / self.cell_area[:,np.newaxis]
        )
        return self.inward_surface_normal
 
    def drag_c(self):
        def pressure_drag_c():
            cell_fp = (
                self.area
                * self.press
                * np.sum(self.normal*self.flow_direction,axis=1)
            )
            cd_p = np.sum(cell_fp,axis=0) * self.const
            return cd_p
        def friction_drag_c():
            cell_ff = self.area * np.sum(
                self.wallshearstress * self.flow_direction,axis=1
            )
            cd_f = np.sum(cell_ff,axis=0) * self.const
            return cd_f
        self.cd_p = pressure_drag_c()
        self.cd_f = friction_drag_c()
        self.cd = self.cd_p + self.cd_f
 
 
    def lift_c(self):
        def pressure_lift_c():
            cell_fp = (
                self.area
                * self.press
                * np.sum(self.inward_surface_normal * self.lift_direction, axis=1)
            )
            cl_p = np.sum(cell_fp, axis=0) * self.const
            return cl_p
 
        def friction_lift_c():
            cell_ff = self.area * np.sum(
                self.wallshearstress * self.lift_direction, axis=1
            )
            cl_f = np.sum(cell_ff, axis=0) * self.const
            return cl_f
 
        self.cl_p = pressure_lift_c()
        self.cl_f = friction_lift_c()
        self.cl = self.cl_p + self.cl_f
 
    def generate_visual_vtk(self):
        cells = [("vertex",np.arange(tuple(self.centroid.shape)[0].reshape(-1,1)))]
        mesh = meshio.Mesh(points=self.centroid,cells=cells)
        mesh.point_data.update({"pressure":self.press})
        mesh.point_data.update({"wallshearstress":self.wallshearstress})
        mesh.point_data.update({"area":self.area})
        mesh.point_data.update({"normal":self.normal})
        meshio.write(os.path.join(self.case_path,f"{self.filename_prefix}.vtk"),mesh)
        return None
 
    def save_values(self):
        os.makedirs(self.save_path,exist_ok=True)
        np.save(
            os.path.join(self.save_path,f"pressure_{self.filename_prefix}.npy"),
            self.press
        )
        np.save(
            os.path.join(self.save_path,f"wallshearstress_{self.filename_prefix}.npy"),
            self.wallshearstress
        )
        np.save(
            os.path.join(self.save_path,f"area_{self.filename_prefix}.npy"),
            self.area
        )
        np.save(
            os.path.join(self.save_path,f"normal_{self.filename_prefix}.npy"),
            self.normal
        )
        np.save(
            os.path.join(self.save_path,f"centroid_{self.filename_prefix}.npy"),
            self.centroid,
        )
        paddle.save(
            obj=self.info,
            path=os.path.join(self.save_path,f"info_{self.filename_prefix}.pdparams")
        )
        return None
 
class ComputeDF:
    def __init__(self,case_path,save_path,caseID,geo="mesh",sdf_spatial_resolution=[64,64,64]):
        self.case_path = case_path
        self.save_path = save_path
        self.filename_prefix = caseID
        # Construct full filename prefix from parent directory name and caseID
        # case_path is like: /path/to/SFE-CR400AF_S-U3-FZ-F087/001
        # parent_dir is: SFE-CR400AF_S-U3-FZ-F087
        # filename_prefix should be: SFE-CR400AF_S-U3-FZ-F087-001
        self.filename_prefix = get_case_filename_prefix(case_path, caseID)
        self.sdf_spatial_resolution = sdf_spatial_resolution
        self.query_points = self.compute_query_point()
        self.geo = geo
 
    def compute_query_point(self,eps=1e-6):
        with open(os.path.join(self.save_path,"global_bounds.txt"),"r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")
            min_bounds = [(float(a) - eps) for a in min_bounds]
            max_bounds = [(float(a) + eps) for a in max_bounds]
        tx = np.linspace(min_bounds[0],max_bounds[0],self.sdf_spatial_resolution[0])
        ty = np.linspace(min_bounds[1],max_bounds[1],self.sdf_spatial_resolution[1])
        tz = np.linspace(min_bounds[2],max_bounds[2],self.sdf_spatial_resolution[2])
 
        query_points = np.stack(np.meshgrid(tx,ty,tz,indexing="ij"),axis=-1).astype(np.float32)
        return query_points
 
    def compute_df_from_mesh(self):
        stl_mesh = o3d.io.read_triangle_mesh(self.case_path + f"/{self.filename_prefix}.stl")
        stl_mesh = o3d.t.geometry.TriangleMesh.from_legacy(stl_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(stl_mesh)
        df = scene.compute_distance(o3d.core.Tensor(self.query_points)).numpy()
        # closest_point = scene.compute_closest_points(o3d.core.Tensor(self.query_points))["points"].numpy()
        df_dict = {
            "df":df,
        }
        return df_dict
 
    def compute_df_from_pcd(self):
        query_points = self.query_points.reshape(-1, 3)
        query_points = o3d.utility.Vector3dVector(query_points)
        pcd_query_points = o3d.geometry.PointCloud()
        pcd_query_points.points = query_points
        train_point = np.load(
            os.path.join(self.save_path, f"centroid_{self.filename_prefix}.npy")
        )
        train_point = o3d.utility.Vector3dVector(train_point)
        pcd_train = o3d.geometry.PointCloud()
        pcd_train.points = train_point
        df = pcd_query_points.compute_point_cloud_distance(pcd_train)
        df = np.asarray(df).reshape(self.sdf_spatial_resolution[0], self.sdf_spatial_resolution[1], self.sdf_spatial_resolution[2])
        closest_point = None
        df_dict = {
            "df": df,
        }
        return df_dict
 
    def save_df(self):
        if self.geo == "mesh":
            df_dict = self.compute_df_from_mesh()
        elif self.geo == "pcd":
            df_dict = self.compute_df_from_pcd()
        else:
            raise ValueError("geo must be 'mesh' or 'pcd'.")
        os.makedirs(self.save_path, exist_ok=True)
        np.save(
            os.path.join(self.save_path, f"df_{self.filename_prefix}.npy"),
            df_dict["df"],
        )
 
 
def categorize_cases(case_names):
    """
    Categorize cases based on filename patterns.
 
    Args:
        case_names: List of case directory names or paths
 
    Returns:
        Dictionary with category counts
    """
    categories = {}
 
    for name in case_names:
        normalized_path = str(name).rstrip('/\\')
        candidates = [os.path.basename(normalized_path)]
 
        parent_name = os.path.basename(os.path.dirname(normalized_path))
        if parent_name and parent_name not in candidates:
            candidates.append(parent_name)
 
        category = "unknown"
        for candidate in candidates:
            # Extract category from pattern: SFE-{category}-U3-FZ-{number}
            # Example: SFE-CR450AF-U3-FZ-001 -> CR450AF
            parts = candidate.split('-')
            if len(parts) >= 4 and parts[0] == "SFE":
                category = parts[1]
                break
 
        categories[category] = categories.get(category, 0) + 1
 
    # Add total count
    categories["total"] = len(case_names)
 
    return categories
 
 
def get_max_workers():
    """
    Get maximum number of worker processes (half of system CPUs).
    Can be overridden via MAX_WORKERS environment variable.
 
    Returns:
        Maximum number of worker processes
    """
    custom_workers = os.environ.get('MAX_WORKERS')
    if custom_workers:
        try:
            return min(int(custom_workers), cpu_count())
        except ValueError:
            pass
 
    cpu_num = cpu_count()
    return max(1, min(cpu_num // 2, 4))
 
 
def get_mp_start_method(stage: str) -> str:
    """
    Select a multiprocessing start method for a processing stage.
 
    Priority:
    1. {STAGE}_MP_START_METHOD environment variable
    2. MP_START_METHOD environment variable
    3. Linux defaults: bounds_computation uses fork, auto_trans uses spawn
    4. spawn on other platforms for compatibility
    """
    stage_env = f"{stage.upper()}_MP_START_METHOD"
    custom_method = os.environ.get(stage_env) or os.environ.get("MP_START_METHOD")
    if custom_method:
        return custom_method
 
    if sys.platform.startswith("linux"):
        if stage == "bounds_computation":
            return "fork"
        if stage == "auto_trans":
            return "spawn"
 
    return "spawn"
 
 
def get_mp_context(stage: str) -> mp.context.BaseContext:
    """Create multiprocessing context for the selected stage."""
    return mp.get_context(get_mp_start_method(stage))
 
 
def data_invalid_check(case_path, save_path, caseID, data_trans: Optional[CFDDataTransiton] = None):
    """Validate a single case before statistics and feature extraction."""
    filename_prefix = get_case_filename_prefix(case_path, caseID)
    try:
        if data_trans is None:
            data_trans = CFDDataTransiton(case_path, save_path, caseID)
        csv_data = data_trans.csv_data
    except Exception as exc:
        return False, f"CFD data load failed: {exc}"
 
    stl_mesh = o3d.io.read_triangle_mesh(os.path.join(case_path, f"{filename_prefix}.stl"))
    if len(stl_mesh.vertices) == 0:
        return False, "stl mesh has 0 points and 0 triangles"
 
    if csv_data.dtype.kind != 'f':
        return False, "non-float data detected"
 
    if np.all(data_trans.wallshearstress == 0):
        return False, "wallshearstress values are all zero"
 
    if np.any(np.abs(data_trans.wallshearstress) > 1e6):
        return False, "wallshearstress values > 1e6"
 
    if np.all(data_trans.press == 0):
        return False, "pressure values are all zero"
 
    if np.any(np.abs(data_trans.press) > 1e6):
        return False, "pressure values > 1e6"
 
    if np.abs(data_trans.cd) > 1.5:
        return False, f"drag coefficient > 1.5, value = {data_trans.cd}"
 
    if np.isinf(data_trans.area).any() or np.isnan(data_trans.area).any():
        return False, "area contains inf/nan"
 
    if np.isinf(data_trans.normal).any() or np.isnan(data_trans.normal).any():
        return False, "normal contains inf/nan"
 
    if np.isinf(data_trans.press).any() or np.isnan(data_trans.press).any():
        return False, "pressure contains inf/nan"
 
    if np.isinf(data_trans.wallshearstress).any() or np.isnan(data_trans.wallshearstress).any():
        return False, "wallshearstress contains inf/nan"
 
    return True, ""
 
# ============ Multiprocessing Worker Functions ============
 
def get_case_filename_prefix(case_path: str, case_id: Optional[str] = None) -> str:
    """Build a stable case filename prefix for logs and generated file names."""
    normalized_path = str(case_path).rstrip('/\\')
    path_basename = os.path.basename(normalized_path)
    parent_dir = os.path.basename(os.path.dirname(normalized_path))
 
    if case_id is None:
        case_id = path_basename
 
    case_id = str(case_id).rstrip('/\\')
 
    if parent_dir.startswith("SFE-"):
        if path_basename == case_id:
            return f"{parent_dir}-{case_id}"
        return parent_dir
 
    return case_id
 
 
def build_streaming_stats(values: np.ndarray) -> Dict[str, Any]:
    """Build count/mean/M2 statistics for a 1D or 2D array."""
    np_values = np.asarray(values, dtype=np.float64)
    if np_values.ndim == 1:
        np_values = np_values[:, np.newaxis]
 
    count = np_values.shape[0]
    mean = np.mean(np_values, axis=0)
    centered = np_values - mean
    m2 = np.sum(centered * centered, axis=0)
    return {
        "count": count,
        "mean": mean,
        "m2": m2,
    }
 
 
 
def merge_streaming_stats(left: Optional[Dict[str, Any]], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two count/mean/M2 statistics dictionaries."""
    if left is None:
        return {
            "count": right["count"],
            "mean": np.array(right["mean"], copy=True),
            "m2": np.array(right["m2"], copy=True),
        }
 
    left_count = left["count"]
    right_count = right["count"]
    total_count = left_count + right_count
    delta = right["mean"] - left["mean"]
    merged_mean = left["mean"] + delta * (right_count / total_count)
    merged_m2 = left["m2"] + right["m2"] + (delta ** 2) * left_count * right_count / total_count
    return {
        "count": total_count,
        "mean": merged_mean,
        "m2": merged_m2,
    }
 
 
def finalize_streaming_stats(stats: Dict[str, Any], value_name: str) -> Tuple[Any, Any]:
    """Convert count/mean/M2 statistics into mean/std."""
    count = stats["count"]
    if count <= 0:
        raise ValueError(f"{value_name} count must be positive")
 
    variance = np.maximum(stats["m2"] / count, 0.0)
    std = np.sqrt(variance)
    mean = stats["mean"]
    return mean.item() if mean.shape == (1,) else mean, std.item() if std.shape == (1,) else std
 
 
def load_case_worker(args: Tuple[str, str, str, bool, bool]) -> Dict[str, Any]:
    """
    Load a single case and optionally compute summary statistics.
 
    The parallel pipeline now treats every successfully loaded case as processable.
    It only rejects cases that cannot be read or transformed into the expected
    in-memory representation.
 
    Args:
        args: Tuple of ``(case_path, save_path, caseID, compute_stats, enable_data_invalid_check)``.
 
    Returns:
        Dictionary containing load status, optional error message, and optional
        per-case statistics used to aggregate global preprocessing bounds.
    """
    case_path, save_path, caseID, compute_stats, enable_data_invalid_check = args
 
    filename_prefix = get_case_filename_prefix(case_path, caseID)
 
    result = {
        "caseID": caseID,
        "filename_prefix": filename_prefix,
        "case_path": case_path,
        "is_loaded": False,
        "load_error": None,
        "load_stage": None,
        "load_traceback": None,
        "stats": None
    }
 
    try:
        result["load_stage"] = "initialize_case"
        data_trans = CFDDataTransiton(case_path, save_path, caseID)
 
        if enable_data_invalid_check:
            result["load_stage"] = "data_invalid_check"
            is_valid, invalid_reason = data_invalid_check(case_path, save_path, caseID, data_trans=data_trans)
            if not is_valid:
                raise ValueError(f"invalid data: {invalid_reason}")
 
        result["load_stage"] = "read_case_arrays"
        area = data_trans.area
        normal = data_trans.normal
        press = data_trans.press
        wss = data_trans.wallshearstress
 
        if compute_stats:
            result["load_stage"] = "compute_case_statistics"
            area_bounds_min = np.min(area)
            area_bounds_max = np.max(area)
            centroid_min = np.min(data_trans.centroid, axis=0)
            centroid_max = np.max(data_trans.centroid, axis=0)

            # 统计的是系数(Cp/Cf)的 mean/std，而非物理量 p/τ。
            # Cp = p / q_ref，Cf = τ / q_ref，q_ref = 0.5*rho*flow_speed**2。
            # flow_speed 使用与训练/推理 compute_q_ref 完全相同的合成来流速度定义：
            #   car_speed 单位为 m/s（原始 JSON 为 km/h，此处 /3.6），
            #   flow_speed = sqrt(vx**2 + vy**2), vx = car + wind*cos(angle), vy = wind*sin(angle)
            c_info = data_trans.info
            car_speed_ms = float(c_info["car_speed"]) / 3.6
            wind_speed = float(c_info["wind_speed"])
            wind_angle_rad = math.radians(float(c_info["wind_angle"]))
            vx = car_speed_ms + wind_speed * math.cos(wind_angle_rad)
            vy = wind_speed * math.sin(wind_angle_rad)
            flow_speed = math.sqrt(vx**2 + vy**2)
            q_ref = 0.5 * float(c_info["density"]) * flow_speed**2

            # 转为系数后再累加统计量
            press_coef = press / q_ref
            wss_coef = wss / q_ref

            result["stats"] = {
                "area_bounds": [area_bounds_min, area_bounds_max],
                "global_bounds": [centroid_min, centroid_max],
                "pressure_stats": build_streaming_stats(press_coef),
                "wss_stats": build_streaming_stats(wss_coef),
                "drag_coefficients": {
                    "cd_p": float(data_trans.cd_p),
                    "cd_f": float(data_trans.cd_f)
                }
            }
 
        result["is_loaded"] = True
        result["load_stage"] = "completed"
 
    except Exception as e:
        result["is_loaded"] = False
        result["stats"] = None
        failed_stage = result["load_stage"] or "unknown_stage"
        result["load_error"] = f"{filename_prefix} failed during {failed_stage}: {str(e)}"
        result["load_traceback"] = traceback.format_exc()
 
    return result
 
def load_cases_and_compute_stats_parallel(
    case_pathes: List[str],
    save_path: str,
    max_workers: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    progress_tracker: Optional[PreprocessProgress] = None,
    enable_data_invalid_check: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Load cases in parallel and aggregate dataset-level statistics.
 
    Despite the legacy naming, this stage no longer performs heuristic legality
    checks. It keeps cases that can be loaded successfully and skips only those
    that fail during data access or feature preparation.
 
    Args:
        case_pathes: List of paths to case directories.
        save_path: Directory to save processed data.
        max_workers: Maximum number of worker processes.
        logger: Logger instance for pipeline logging.
        progress_tracker: Optional progress tracker for real-time JSON updates.
 
    Returns:
        Tuple of ``(loaded_case_paths, stats_dict)`` where ``stats_dict`` contains
        global bounds, mean/std values, and per-case drag coefficients.
    """
    if logger is None:
        logger = logging.getLogger()
 
    os.makedirs(save_path, exist_ok=True)
 
    if max_workers is None:
        max_workers = get_max_workers()
 
    max_workers = max(1, min(max_workers, len(case_pathes))) if case_pathes else 1
    mp_context = get_mp_context("bounds_computation")
    start_method = mp_context.get_start_method()
 
    logger.info(f"Starting parallel case loading and stats computation with {max_workers} workers...")
    logger.info(f"Using multiprocessing start method '{start_method}' for case loading")
 
    stats_args = [
        (cp, save_path, os.path.basename(cp.rstrip('/')), True, enable_data_invalid_check)
        for cp in case_pathes
    ]
 
    loaded_case_paths = []
    skipped_case_names = []
    stats_by_case = {}
 
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor, \
         tqdm(total=len(case_pathes), desc="Loading cases", unit="case") as pbar:
 
        future_to_case = {
            executor.submit(load_case_worker, args): {
                "case_id": args[2],
                "filename_prefix": get_case_filename_prefix(args[0], args[2]),
            }
            for args in stats_args
        }
 
        for future in as_completed(future_to_case):
            case_info = future_to_case[future]
            filename_prefix = case_info["filename_prefix"]
            try:
                result = future.result()
            except Exception as e:
                skipped_case_names.append(filename_prefix)
                logger.error(f"{filename_prefix}: worker failed with exception: {e}")
                pbar.update(1)
                continue
 
            if result["is_loaded"]:
                loaded_case_paths.append(result["case_path"])
                if result["stats"] is not None:
                    # 用唯一的 case_path 作为 key，避免不同父目录下同名子目录
                    # （如多个父目录各有 "001"）相互覆盖导致统计 case 丢失。
                    stats_by_case[result["case_path"]] = result["stats"]
                logger.info(f"{result['filename_prefix']}: case loaded")
            else:
                skipped_case_names.append(result["filename_prefix"])
                logger.error(f"{result['filename_prefix']}: {result['load_error']}")
                if result.get("load_traceback"):
                    logger.debug(result["load_traceback"])
 
            if progress_tracker is not None:
                processed_cases = len(loaded_case_paths) + len(skipped_case_names)
                progress_tracker.update_bounds_computation_progress(processed_cases, len(case_pathes))
                progress_tracker.update_bounds_computation_results(
                    len(loaded_case_paths),
                    len(skipped_case_names),
                    skipped_case_names.copy(),
                )
 
            pbar.update(1)
 
    logger.info(f"Bounds computation complete: {len(loaded_case_paths)} loaded, {len(skipped_case_names)} skipped")
 
    if len(loaded_case_paths) == 0:
        raise ValueError("All cases failed to load. No valid data to process.")
 
    logger.info(f"Statistics were collected during case loading for {len(stats_by_case)} loaded cases.")
 
    area_bounds_all_min = []
    area_bounds_all_max = []
    global_bounds_all_min = []
    global_bounds_all_max = []
 
    pressure_stats = None
    wss_stats = None
 
    drag_coefficients = {}
 
    ordered_loaded_case_paths = []
    for case_path in loaded_case_paths:
        case_id = os.path.basename(case_path.rstrip('/'))
        filename_prefix = get_case_filename_prefix(case_path, case_id)
        stats = stats_by_case.get(case_path)
        if stats is None:
            logger.warning(f"{filename_prefix}: statistics missing after loading, skipping from aggregation")
            continue
 
        stats_values = [
            stats["area_bounds"][0], stats["area_bounds"][1],
            stats["global_bounds"][0], stats["global_bounds"][1],
            stats["pressure_stats"]["mean"], stats["pressure_stats"]["m2"],
            stats["wss_stats"]["mean"], stats["wss_stats"]["m2"],
            stats["drag_coefficients"]["cd_p"], stats["drag_coefficients"]["cd_f"],
        ]
        if any(not np.all(np.isfinite(np.asarray(value))) for value in stats_values):
            logger.warning(f"{filename_prefix}: NaN/Inf found in statistics, skipping from aggregation")
            continue
 
        wss_std = np.sqrt(np.maximum(np.asarray(stats['wss_stats']['m2']) / stats['wss_stats']['count'], 0.0))
        logger.info(
            f"{filename_prefix}: "
            f"wss_abs_mean_max={np.max(np.abs(stats['wss_stats']['mean'])):.4e}, "
            f"wss_mean={np.array2string(np.asarray(stats['wss_stats']['mean']), precision=4)}, "
            f"wss_std={np.array2string(wss_std, precision=4)}"
        )
 
        ordered_loaded_case_paths.append(case_path)
        drag_coefficients[filename_prefix] = stats["drag_coefficients"]
 
        area_bounds_all_min.append(stats["area_bounds"][0])
        area_bounds_all_max.append(stats["area_bounds"][1])
        global_bounds_all_min.append(stats["global_bounds"][0])
        global_bounds_all_max.append(stats["global_bounds"][1])
 
        pressure_stats = merge_streaming_stats(pressure_stats, stats["pressure_stats"])
        wss_stats = merge_streaming_stats(wss_stats, stats["wss_stats"])
 
    loaded_case_paths = ordered_loaded_case_paths
 
    if len(loaded_case_paths) == 0:
        raise ValueError("All loaded cases were skipped during statistics aggregation.")
 
    area_bounds = [np.min(area_bounds_all_min), np.max(area_bounds_all_max)]
    global_bounds = [np.min(np.array(global_bounds_all_min), axis=0), np.max(np.array(global_bounds_all_max), axis=0)]
    if not np.all(np.isfinite(np.asarray(area_bounds))) or not np.all(np.isfinite(np.asarray(global_bounds))):
        raise ValueError("NaN/Inf found in aggregated bounds.")
 
    train_pressure_mean, train_pressure_std = finalize_streaming_stats(pressure_stats, "pressure")
    train_wss_mean, train_wss_std = finalize_streaming_stats(wss_stats, "wallshearstress")
    if not np.all(np.isfinite(np.asarray([train_pressure_mean, train_pressure_std]))) or not np.all(np.isfinite(np.asarray(train_wss_mean))) or not np.all(np.isfinite(np.asarray(train_wss_std))):
        raise ValueError("NaN/Inf found in aggregated mean/std.")
 
    stats_dict = {
        "area_bounds": area_bounds,
        "global_bounds": global_bounds,
        "train_pressure_coef_mean_std": [train_pressure_mean, train_pressure_std],
        "train_wallshearstress_coef_mean_std": [train_wss_mean, train_wss_std],
        "drag_coefficients": drag_coefficients
    }
 
    # Save bounds files
    for k, v in stats_dict.items():
        if k == "drag_coefficients":
            continue  # Drag coefficients handled separately
        with open(save_path + f"/{k}.txt", "w") as f:
            if k == "global_bounds" or k == "train_wallshearstress_coef_mean_std":
                for i in range(len(v)):
                    f.write(" ".join(str(number) for number in v[i].tolist()) + '\n')
            else:
                for i in range(len(v)):
                    f.write("%s\n" % v[i].tolist() if isinstance(v[i], np.ndarray) else "%s\n" % v[i])
 
    # Save drag coefficients
    if drag_coefficients:
        sorted_drag_coefficients = dict(sorted(
            drag_coefficients.items(),
            key=lambda x: x[1]["cd_p"]
        ))
 
        log_dir = os.path.join(save_path, "log")
        os.makedirs(log_dir, exist_ok=True)
 
        drag_coeff_json_file = os.path.join(log_dir, "drag_coefficients.json")
        with open(drag_coeff_json_file, "w", encoding="utf-8") as f:
            json.dump(sorted_drag_coefficients, f, indent=2, ensure_ascii=False)
 
        drag_coeff_md_file = os.path.join(log_dir, "drag_coefficients.md")
        with open(drag_coeff_md_file, "w", encoding="utf-8") as f:
            f.write("| caseid | cd_p | cd_f |\n")
            f.write("|--------|------|------|\n")
            for case_id, coeffs in sorted_drag_coefficients.items():
                f.write(f"| {case_id} | {coeffs['cd_p']:.6f} | {coeffs['cd_f']:.6f} |\n")
 
        logger.info(f"Drag coefficients saved to {drag_coeff_json_file} and {drag_coeff_md_file}")
        logger.info(f"Total {len(sorted_drag_coefficients)} cases with drag coefficients collected")
 
    logger.info(f"Statistics computed and saved: {len(loaded_case_paths)} cases included")
    logger.info(f"Area bounds: [{area_bounds[0]:.4e}, {area_bounds[1]:.4e}]")
    logger.info(f"Pressure coef mean/std: [{train_pressure_mean:.4f}, {train_pressure_std:.4f}]")
 
    return loaded_case_paths, stats_dict
 
 
def auto_trans_worker(args: Tuple[str, str, str, List[int]]) -> Tuple[str, str, bool, str]:
    """
    Worker function for processing a single case in auto_trans.
 
    Args:
        args: Tuple of (case_path, save_path, caseID, sdf_spatial_resolution)
 
    Returns:
        Tuple of (caseID, filename_prefix, success, error_message)
    """
    import traceback
 
    case_path, save_path, caseID, sdf_spatial_resolution = args
 
    filename_prefix = get_case_filename_prefix(case_path, caseID)
 
    try:
        data_trans = CFDDataTransiton(case_path, save_path, caseID)
        data_trans.save_values()
        mesh_trans = ComputeDF(case_path, save_path, caseID, sdf_spatial_resolution=sdf_spatial_resolution)
        mesh_trans.save_df()
        return (caseID, filename_prefix, True, "")
    except Exception as e:
        error_msg = f"{filename_prefix} auto_trans failed: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        print(f"[ERROR] Traceback: {traceback.format_exc()}", file=sys.stderr)
        return (caseID, filename_prefix, False, str(e))
 
 
def auto_trans_parallel(
    case_pathes: List[str],
    save_path: str,
    sdf_spatial_resolution: List[int],
    max_workers: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    progress_tracker: Optional[PreprocessProgress] = None,
) -> Tuple[int, int, List[str]]:
    """
    Process all cases in parallel using auto_trans.
 
    Args:
        case_pathes: List of paths to case directories
        save_path: Directory to save processed data
        sdf_spatial_resolution: SDF spatial resolution [x, y, z]
        max_workers: Maximum number of worker processes (default: half of system CPUs)
        logger: Logger instance for logging
        progress_tracker: Optional progress tracker for real-time JSON updates.
 
    Returns:
        Tuple of (success_count, failure_count, failed_case_names)
    """
    if logger is None:
        logger = logging.getLogger()
 
    if max_workers is None:
        max_workers = get_max_workers()
 
    max_workers = max(1, min(max_workers, len(case_pathes))) if case_pathes else 1
    logger.info(f"Starting parallel auto_trans with {max_workers} workers...")
 
    mp_context = get_mp_context("auto_trans")
    start_method = mp_context.get_start_method()
    logger.info(f"Using multiprocessing start method '{start_method}' for auto_trans")
 
    args_list = [(cp, save_path, os.path.basename(cp.rstrip('/')), sdf_spatial_resolution) for cp in case_pathes]
 
    success_count = 0
    failure_count = 0
    failed_case_names = []
 
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor, \
         tqdm(total=len(case_pathes), desc="Processing cases (auto_trans)", unit="case") as pbar:
 
        future_to_case = {
            executor.submit(auto_trans_worker, args): {
                "case_id": args[2],
                "filename_prefix": get_case_filename_prefix(args[0], args[2]),
            }
            for args in args_list
        }
 
        for future in as_completed(future_to_case):
            case_info = future_to_case[future]
            case_id = case_info["case_id"]
            filename_prefix = case_info["filename_prefix"]
            try:
                _finished_case_id, filename_prefix, success, error_msg = future.result()
            except Exception as e:
                failure_count += 1
                failed_case_names.append(filename_prefix)
                logger.error(f"Extract features of {filename_prefix} occurred error: {e}")
                if progress_tracker is not None:
                    processed_cases = success_count + failure_count
                    progress_tracker.update_feature_extraction_progress(
                        filename_prefix,
                        processed_cases,
                        len(case_pathes),
                    )
                pbar.update(1)
                continue
 
            if success:
                success_count += 1
                logger.info(f"({success_count}/{len(case_pathes)}) {filename_prefix} has been processed successfully.")
            else:
                failure_count += 1
                failed_case_names.append(filename_prefix)
                logger.error(f"Extract features of {filename_prefix} occurred error: {error_msg}")
 
            if progress_tracker is not None:
                processed_cases = success_count + failure_count
                progress_tracker.update_feature_extraction_progress(
                    filename_prefix,
                    processed_cases,
                    len(case_pathes),
                )
 
            pbar.update(1)
 
    logger.info(f"Auto_trans complete: {success_count} succeeded, {failure_count} failed")
    if failed_case_names:
        logger.warning(f"Failed cases: {failed_case_names}")
 
    return success_count, failure_count, failed_case_names
 
 
# ============ Legacy Functions (kept for compatibility) ============
 
def auto_trans(case_path,save_path,caseID,sdf_spatial_resolution):
    """Legacy function - kept for compatibility but should use auto_trans_parallel for parallel processing."""
    data_trans = CFDDataTransiton(case_path,save_path,caseID)
    data_trans.save_values()
    mesh_trans = ComputeDF(case_path, save_path, caseID, sdf_spatial_resolution=sdf_spatial_resolution)
    mesh_trans.save_df()
 
def load_case_check(case_path,save_path,caseID):
    """
    Legacy compatibility wrapper for the old single-case load-check interface.
 
    The helper is kept for callers that still want a boolean loadability check
    around ``CFDDataTransiton``.
    """
    logger = logging.getLogger()
    filename_prefix = get_case_filename_prefix(case_path, caseID)
    try:
        data_trans = CFDDataTransiton(case_path,save_path,caseID)
    except Exception:
        logger.error(f"{filename_prefix} CFD data load failed.")
        return False
 
    logger.info(
        f"{filename_prefix}: data load passed — "
        f"{len(data_trans.area)} cells, press_range=[{data_trans.press.min():.4f}, {data_trans.press.max():.4f}], "
        f"area_range=[{data_trans.area.min():.4e}, {data_trans.area.max():.4e}]"
    )
    return True
 
 
def print_floats(*args):
    """Log floating-point values with a fixed four-decimal format."""
    format_str = "{:." + str(4) + "f}"
    for num in args:
        logging.info(format_str.format(num) + "\t")
    logging.info("\n")
 
 
def extract_number(s):
    """Extract the first integer from the first four characters of a string."""
    s = s[:4]
    ids = re.findall('\\d+',s)
    return int(ids[0])
 
 
def failed_caseid(failed_case_pathes):
    """Convert failed case paths into their case-id list."""
    failed_case_ids = []
    for path in failed_case_pathes:
        case_id = os.path.basename(path.rstrip('/'))
        failed_case_ids.append(case_id)
    return failed_case_ids
 
 
def resolve_case_paths(pre_input_path: Any) -> List[str]:
    """Expand the configured input root into a concrete list of case directories."""
    if isinstance(pre_input_path, str):
        return [
            os.path.join(pre_input_path, directory)
            for directory in os.listdir(pre_input_path)
        ]
    if isinstance(pre_input_path, ListConfig):
        return list(pre_input_path)
    if isinstance(pre_input_path, list):
        return pre_input_path
    raise ValueError("cfg.pre_input_path must be str or list")
 
 
def setup_preprocess_logging(pre_output_path: str) -> logging.Logger:
    """Configure file and console logging for the preprocessing run."""
    os.makedirs(os.path.join(pre_output_path, "log"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(filename)s][%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(f"{pre_output_path}/log/trainDataPreprocess.txt", mode='w'),
            logging.StreamHandler()
        ],
        force=True,
    )
    return logging.getLogger()
 
 
def initialize_preprocess_run(
    cfg: DictConfig,
    pre_output_path: str,
    pre_input_pathes: List[str],
) -> Tuple[PreprocessProgress, logging.Logger, int]:
    """Initialize logging, progress tracking, and run-level metadata."""
    logger = setup_preprocess_logging(pre_output_path)
    progress_tracker = PreprocessProgress(pre_output_path)
    max_workers = get_max_workers()
    config_dict = {
        "pre_input_path": list(cfg.pre_input_path) if isinstance(cfg.pre_input_path, ListConfig) else cfg.pre_input_path,
        "pre_output_path": cfg.pre_output_path,
        "process_mode": cfg.process_mode,
        "enable_data_invalid_check": cfg.get("enable_data_invalid_check", True),
        "sdf_spatial_resolution": list(cfg.sdf_spatial_resolution),
        "fno_modes": list(cfg.fno_modes),
        "parallel_enabled": True,
        "max_workers": max_workers
    }
    progress_tracker.update_config(config_dict)
    logger.info(" ============================ TRAINDATAPREPROCESS START ========================== ")
    logger.info(f"Preprocess config: \n{OmegaConf.to_yaml(cfg)}\n")
    logger.info(f"All {len(pre_input_pathes)} cases have been detected.\n")
    logger.info(f"Parallel processing enabled with max_workers={max_workers}\n")
 
    return progress_tracker, logger, max_workers
 
 
def collect_case_statistics(
    pre_input_pathes: List[str],
    pre_output_path: str,
    progress_tracker: PreprocessProgress,
    logger: logging.Logger,
) -> Dict[str, int]:
    """Summarize case-name distribution and persist the statistics file."""
    case_names = [os.path.basename(path.rstrip('/')) for path in pre_input_pathes]
    case_statistics = categorize_cases(pre_input_pathes)
 
    stats_file = os.path.join(pre_output_path, "log", "case_statistics.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(case_statistics, f, indent=2, ensure_ascii=False)
 
    logger.info(f"Case statistics saved to {stats_file}")
    logger.info(f"Case distribution: {case_statistics}")
 
    progress_tracker.update_case_statistics(case_statistics, len(pre_input_pathes))
    progress_tracker.update_output_files("case_statistics", stats_file)
    return case_statistics
 
 
def load_cases_for_preprocess(
    pre_input_pathes: List[str],
    pre_output_path: str,
    progress_tracker: PreprocessProgress,
    logger: logging.Logger,
    max_workers: int,
    enable_data_invalid_check: bool,
) -> Tuple[List[str], Dict[str, Any], List[str], float]:
    """Load cases in parallel, collect statistics, and record skipped inputs."""
    progress_tracker.start_stage("bounds_computation", "Computing global bounds and statistics", len(pre_input_pathes))
    progress_tracker.update_bounds_computation_progress(0, len(pre_input_pathes))
 
    logger.info(' ============================ BOUNDS COMPUTATION START ======================== ')
    logger.info(f"** Starting bounds computation for {len(pre_input_pathes)} raw cases...")
 
    t1 = default_timer()
    loaded_case_paths, stats_dict = load_cases_and_compute_stats_parallel(
        pre_input_pathes,
        pre_output_path,
        max_workers=max_workers,
        logger=logger,
        progress_tracker=progress_tracker,
        enable_data_invalid_check=enable_data_invalid_check,
    )
    elapsed = default_timer() - t1
 
    skipped_case_names = [get_case_filename_prefix(cp) for cp in pre_input_pathes if cp not in loaded_case_paths]
    loaded_cases = len(loaded_case_paths)
    skipped_cases = len(skipped_case_names)
 
    progress_tracker.update_bounds_computation_results(loaded_cases, skipped_cases, skipped_case_names)
    progress_tracker.end_stage(
        "bounds_computation",
        f"Bounds computation completed: {loaded_cases} loaded, {skipped_cases} skipped"
    )
 
    logger.info(f"** {skipped_cases} cases were skipped among {len(pre_input_pathes)} detected cases.")
    logger.info(f"bounds computation has been done, elapsed time: {elapsed:.2f} s.")
    logger.info('============================= BOUNDS COMPUTATION END ===========================\n')
    return loaded_case_paths, stats_dict, skipped_case_names, elapsed
 
 
def finalize_preprocess_outputs(progress_tracker: PreprocessProgress, pre_output_path: str) -> None:
    """Register generated summary files in the progress tracker."""
    drag_json_file = os.path.join(pre_output_path, "log", "drag_coefficients.json")
    drag_md_file = os.path.join(pre_output_path, "log", "drag_coefficients.md")
    if os.path.exists(drag_json_file):
        progress_tracker.update_output_files("drag_coefficients_json", drag_json_file)
    if os.path.exists(drag_md_file):
        progress_tracker.update_output_files("drag_coefficients_md", drag_md_file)
 
    bounds_files = [
        "global_bounds.txt",
        "area_bounds.txt",
        "train_pressure_coef_mean_std.txt",
        "train_wallshearstress_coef_mean_std.txt"
    ]
    for file_name in bounds_files:
        file_path = os.path.join(pre_output_path, file_name)
        if os.path.exists(file_path):
            progress_tracker.update_output_files("bounds_files", file_path)
 
 
def run_train_preprocess(cfg: DictConfig) -> None:
    """Execute the full train-mode preprocessing pipeline."""
    sdf_spatial_resolution = cfg.sdf_spatial_resolution
    cfg.fno_modes = [r // 2 for r in sdf_spatial_resolution]
    pre_output_path = cfg.pre_output_path
    enable_data_invalid_check = cfg.get("enable_data_invalid_check", True)
    pre_input_pathes = resolve_case_paths(cfg.pre_input_path)
    progress_tracker, logger, max_workers = initialize_preprocess_run(cfg, pre_output_path, pre_input_pathes)
 
    collect_case_statistics(pre_input_pathes, pre_output_path, progress_tracker, logger)
 
    loaded_case_paths, _stats_dict, skipped_case_names, bounds_elapsed = load_cases_for_preprocess(
        pre_input_pathes,
        pre_output_path,
        progress_tracker,
        logger,
        max_workers,
        enable_data_invalid_check,
    )
    loaded_cases = len(loaded_case_paths)
 
    logging.info('=========================== FEATURE EXTRACTION START ========================')
    t3 = default_timer()
    if loaded_cases == 0:
        logging.info(f"Skipped cases: {skipped_case_names}")
        logging.info("No loadable cases were found, terminate preprocess.")
        progress_tracker.update_bounds_computation_results(0, len(skipped_case_names), skipped_case_names)
        progress_tracker.complete()
        return
 
    if skipped_case_names:
        logging.info(f"** Skipped cases: {skipped_case_names}")
 
    logging.info("** Summary statistics has been computed and saved successfully.\n")
 
    progress_tracker.start_stage("feature_extraction", "Extracting features from loaded cases", loaded_cases)
    progress_tracker.update_feature_extraction_progress("", 0, loaded_cases)
    logging.info(f"** Starting to extract features for {loaded_cases}(loaded) / {len(pre_input_pathes)}(all) cases...")
 
    success_count, failure_count, failed_case_names = auto_trans_parallel(
        loaded_case_paths,
        pre_output_path,
        sdf_spatial_resolution,
        max_workers=max_workers,
        logger=logger,
        progress_tracker=progress_tracker,
    )
 
    logging.info('** Feature Extraction have been done.')
    logging.info('\n')
    t4 = default_timer()
    logging.info(f"Feature extraction has been done, elapsed time: {t4 - t3:.2f} s.")
 
    if failed_case_names:
        logging.info(f"** Failed cases: {failed_case_names}")
 
    logging.info(f"** All {success_count}(loaded) / {len(pre_input_pathes)}(all) cases have been processed successfully.")
    logging.info(f"** Feature extraction failures: {failure_count}")
 
    progress_tracker.end_stage("feature_extraction", f"Feature extraction completed for {success_count} cases")
    progress_tracker.complete()
    finalize_preprocess_outputs(progress_tracker, pre_output_path)
 
    logging.info('=========================== FEATURE EXTRACTION END ==========================')
    logging.info(f"train data preprocess has been done, elapsed time: {t4 - t3 + bounds_elapsed:.2f} s.")
 
 
@hydra.main(config_path="./configs",config_name="train.yaml")
def main(cfg:DictConfig)->None:
    """Hydra entry point for the parallel training-data preprocessing pipeline."""
    if cfg.process_mode != "train":
        raise ValueError("当前 process_mode 不是 train，请检查 Hydra 配置文件中的 process_mode 变量")
    run_train_preprocess(cfg)
 
 
if __name__ == '__main__':
    main()