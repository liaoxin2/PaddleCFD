import json
import logging
import os
import sys
from timeit import default_timer
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import hydra
import meshio
import numpy as np
import paddle
import pyvista as pv
import vtk
import math
from omegaconf import DictConfig
from paddle import distributed as dist
from paddle.distributed import ParallelEnv
from paddle.distributed import fleet
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

from ppcfd.models.ppfno.data import instantiate_inferencedatamodule
from ppcfd.models.ppfno.data.datamodule_lazy import compute_q_ref
from ppcfd.models.ppfno.losses import LpLoss
from ppcfd.models.ppfno.networks import instantiate_network
from ppcfd.models.ppfno.optim.schedulers import instantiate_scheduler
from ppcfd.models.ppfno.utils.average_meter import AverageMeter
from ppcfd.models.ppfno.utils.average_meter import AverageMeterDict
from ppcfd.models.ppfno.utils.dot_dict import DotDict
from ppcfd.models.ppfno.utils.dot_dict import flatten_dict


def set_seed(seed: int = 0):
    paddle.seed(seed=seed)
    np.random.seed(seed)


world_size = dist.get_world_size()
if world_size > 1:
    strategy = fleet.DistributedStrategy()
    strategy.find_unused_parameters = True
    fleet.init(is_collective=True, strategy=strategy)


def save_vtp_from_dict(
    filename: str,
    data_dict: Dict[str, np.ndarray],
    coord_keys: Tuple[str, ...],
    value_keys: Tuple[str, ...],
    num_timestamps: int = 1,
):

    if len(coord_keys) not in [3]:
        raise ValueError(f"ndim of coord ({len(coord_keys)}) should be 3 in vtp format")

    coord = [data_dict[k] for k in coord_keys if k not in ("t", "sdf")]
    assert all([c.ndim == 2 for c in coord]), "array of each axis should be [*, 1]"
    coord = np.concatenate(coord, axis=1)

    if not isinstance(coord, np.ndarray):
        raise ValueError(f"type of coord({type(coord)}) should be ndarray.")
    if len(coord) % num_timestamps != 0:
        raise ValueError(
            f"coord length({len(coord)}) should be an integer multiple of "
            f"num_timestamps({num_timestamps})"
        )
    if coord.shape[1] not in [3]:
        raise ValueError(f"ndim of coord({coord.shape[1]}) should be 3 in vtp format.")

    if len(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    npoint = len(coord)
    nx = npoint // num_timestamps
    if filename.endswith(".vtp"):
        filename = filename[:-4]

    for t in range(num_timestamps):
        coord_ = coord[t * nx : (t + 1) * nx]
        point_cloud = pv.PolyData(coord_)
        for k in value_keys:
            value_ = data_dict[k][t * nx : (t + 1) * nx]
            if value_ is not None and not isinstance(value_, np.ndarray):
                raise ValueError(f"type of value({type(value_)}) should be ndarray.")
            if value_ is not None and len(coord_) != len(value_):
                raise ValueError(
                    f"coord length({len(coord_)}) should be equal to value length({len(value_)})"
                )
            point_cloud[k] = value_

        if num_timestamps > 1:
            width = len(str(num_timestamps - 1))
            point_cloud.save(f"{filename}_t-{t:0{width}}.vtp", binary=True)
        else:
            point_cloud.save(f"{filename}.vtp", binary=True)

    if num_timestamps > 1:
        logging.info(
            f"Visualization results are saved to: {filename}_t-{0:0{width}}.vtp ~ "
            f"{filename}_t-{num_timestamps - 1:0{width}}.vtp"
        )
    else:
        logging.info(f"Visualization result is saved to: {filename}.vtp")


@paddle.no_grad()
def inference(cfg: DictConfig):

    os.makedirs(cfg.reason_output_path, exist_ok=True)
    os.makedirs(os.path.join(cfg.reason_output_path, "log"), exist_ok=True)
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # init logger
    logging.basicConfig(
        filename=os.path.join(cfg.reason_output_path, "log", "reason.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: %(message)s",
        force=True,
    )

    inference_json_file_path = os.path.join(
        cfg.reason_output_path,
        "json",
        "reason.json",
    )

    def create_json(json_file_path):
        if not os.path.exists(os.path.dirname(json_file_path)):
            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

        if os.path.isfile(json_file_path):
            os.remove(json_file_path)

        with open(json_file_path, "w") as file:
            json.dump([], file)

    create_json(inference_json_file_path)

    def append_dict_to_json_list(file_path, dict_element):
        with open(file_path, "r") as file:
            data = json.load(file)

        if isinstance(data, list):
            data.append(dict_element)
        else:
            print("Error: The root of the JSON file is not a list.")
            return
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    # init model
    model = instantiate_network(cfg)
    loss_fn = LpLoss(size_average=True)
    if cfg.enable_ddp:
        model = fleet.distributed_model(model)
    assert cfg.state is not None, "checkpoint must be given."
    state = paddle.load(path=str(cfg.state))
    model.set_state_dict(state_dict=state["model"])

    device = ParallelEnv().device_id
    memory_allocated = paddle.device.cuda.memory_allocated(device=device) / (
        1024 * 1024 * 1024
    )
    logging.info(f"Memory usage with model loading: {memory_allocated:.2f} GB")

    # run prediction
    datamodule = instantiate_inferencedatamodule(
        cfg, cfg.reason_input_path, cfg.pre_output_path, cfg.n_inference_num
    )
    inference_dataloader = datamodule.inference_dataloader(
        enable_ddp=cfg.enable_ddp, batch_size=cfg.batch_size
    )

    logging.info(f"Start evaluting {cfg.model} ...")

    if isinstance(model, paddle.DataParallel):
        model = model._layers
    model.eval()
    eval_meter = AverageMeterDict()
    visualize_data_dicts = []

    def cal_mre(pred, label):
        return paddle.abs(x=pred - label) / paddle.abs(x=label)

    for i, data_dict in enumerate(inference_dataloader):
        if ',' in data_dict['info'][0]['wind_speed']:
            value_list = [float(wind_speed) for wind_speed in data_dict['info'][0]['wind_speed'].split(',')]
            value_type = 'wind_speed'
        else:
            value_list = [float(wind_angle) for wind_angle in data_dict['info'][0]['wind_angle'].split(',')]
            value_type = 'wind_angle'

        for value in value_list:
            inference_json_dict = {}
            if value_type == 'wind_speed':
                data_dict['info'][0]['wind_speed'] = value
                inference_json_dict['type'] = 'wind_speed'
            else:
                data_dict['info'][0]['wind_angle'] = value
                inference_json_dict['type'] = 'wind_angle'

            # 扫描值已写回 info，须基于当前 wind 值重算 q_ref，
            # 保证 decode 用的 q_ref 与积分用的 F_const 满足 F_const == 1/(q_ref*A)。
            data_dict["q_ref"] = [compute_q_ref(data_dict["info"][0])]


            device = ParallelEnv().device_id
            device = paddle.CUDAPlace(device)
            try:
                t1 = default_timer()
                out_dict, pred, F_M_dict = model.inference_dict(
                    device, data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode
                )
                t2 = default_timer()
                logging.info(f"Inference {i} costs: {t2 - t1:.2f} seconds.")
                # print('cd_dict:', cd_dict)
                if cfg.save_eval_results:
                    save_eval_results(
                        cfg,
                        pred,
                        value,
                        datamodule.inference_indices[0],
                        datamodule.inference_full_caseids[0],
                        decode_fn=datamodule.decode,
                        q_ref=data_dict["q_ref"][0],
                    )
                # paddle.device.cuda.empty_cache()
            except MemoryError as e:
                if "Out of memory" in str(e):
                    logging.info(f"WARNING: OOM on sample {i}, skipping this sample.")
                    if hasattr(paddle.device.cuda, "empty_cache"):
                        paddle.device.cuda.empty_cache()
                    continue
                else:
                    raise
            msg = f"Eval sample {i}... L2_Error: "
            for k, v in out_dict.items():
                if k.split("_")[0] == "L2":
                    msg += f"{k}: {v.item():.4f}, "
                    eval_meter.update({k: v})
            msg += f"|| MRE and Value: "

            # 真·系数空间：F_pred_modify/M_pred_modify 为无量纲系数，折回物理力/力矩用于日志
            F_pred_modify = F_M_dict["F_pred_modify"] / F_M_dict["F_const"]
            M_pred_modify = F_M_dict["M_pred_modify"] / F_M_dict["M_const"]
            F_pred = out_dict["F_pred"]
            M_pred = out_dict["M_pred"]
            eval_meter.update({"F_pred_modify": F_pred_modify})
            eval_meter.update({"M_pred_modify": M_pred_modify})
            msg += f"F_pred_modify: {F_pred_modify.numpy()}, "
            msg += f"M_pred_modify: {M_pred_modify.numpy()}, "


            load_types = {
                'aerodynamic_lift': {'pred': F_M_dict['F_pred'][1]},
                'aerodynamic_drag': {'pred': F_M_dict['F_pred'][0]},
                'pneumatic_lateral_force': {'pred': F_M_dict['F_pred'][2]},
                'pneumatic_overturning_moment': {'pred': F_M_dict['M_pred'][0]},
                'pneumatic_pitching_moment': {'pred': F_M_dict['M_pred'][1]},
                'pneumatic_roll_moment': {'pred': F_M_dict['M_pred'][2]},
            }
            load_types_modify = {
                'aerodynamic_lift': {'pred': F_M_dict['F_pred_modify'][1]},
                'aerodynamic_drag': {'pred': F_M_dict['F_pred_modify'][0]},
                'pneumatic_lateral_force': {'pred': F_M_dict['F_pred_modify'][2]},
                'pneumatic_overturning_moment': {'pred': F_M_dict['M_pred_modify'][0]},
                'pneumatic_pitching_moment': {'pred': F_M_dict['M_pred_modify'][1]},
                'pneumatic_roll_moment': {'pred': F_M_dict['M_pred_modify'][2]},
            }

            inference_json_dict['car_speed'] = data_dict["info"][0]["car_speed"]
            inference_json_dict['wind_speed'] = data_dict["info"][0]["wind_speed"]
            inference_json_dict['wind_angle'] = data_dict["info"][0]["wind_angle"]

            mass_density = float(data_dict["info"][0]["density"])
            reference_area = float(data_dict["info"][0]["area"])
            typical_length = float(data_dict["info"][0]["typical_length"])
            car_speed = float(data_dict["info"][0]["car_speed"])
            wind_speed = float(data_dict["info"][0]["wind_speed"])
            wind_angle_rad = math.radians(float(data_dict["info"][0]["wind_angle"]))
            vx = car_speed + wind_speed * math.cos(wind_angle_rad)
            vy = wind_speed * math.sin(wind_angle_rad)
            flow_speed = math.sqrt(vx**2 + vy**2)
            F_const = 2.0 / (mass_density * flow_speed**2 * reference_area)
            M_const = 2.0 / (mass_density * flow_speed**2 * reference_area * typical_length)

            for load_name, values in load_types.items():
                cal_val = values['pred'].numpy() if hasattr(values['pred'], 'numpy') else values['pred']
                cal_val_modify = load_types_modify[load_name]['pred'].numpy() if hasattr(load_types_modify[load_name]['pred'], 'numpy') else load_types_modify[load_name]['pred']
                # 真·系数空间：修正网络直接输出无量纲系数，故 modify 分支的物理值
                # 需 ×(1/const) 还原；no_modify 分支仍是骨干积分的物理力，×const 得系数。
                if load_name in ['aerodynamic_lift', 'aerodynamic_drag', 'pneumatic_lateral_force']:
                    inference_json_dict[load_name] = {
                        'cal_value': float(cal_val_modify)/F_const,
                        'coefficient': float(cal_val_modify),
                        'cal_value_no_modify': float(cal_val),
                        'coefficient_no_modify': float(cal_val)*F_const,
                    }
                else:
                    inference_json_dict[load_name] = {
                        'cal_value': float(cal_val_modify)/M_const,
                        'coefficient': float(cal_val_modify),
                        'cal_value_no_modify': float(cal_val),
                        'coefficient_no_modify': float(cal_val)*M_const,
                    }

            append_dict_to_json_list(inference_json_file_path, inference_json_dict)

            logging.info(msg)

    t3 = default_timer()
    msg = (
        f"Inference + vtp file saving took {t3 - t1:.2f} seconds. Everage eval values: "
    )
    eval_dict = eval_meter.avg
    for k, v in eval_dict.items():
        msg += f"{v}({k}), "
    logging.info(msg)
    max_memory_allocated = paddle.device.cuda.max_memory_allocated(device=device) / (
        1024 * 1024 * 1024
    )
    logging.info(f"Memory Usage: {max_memory_allocated:.2f} GB (MAX).")


def save_eval_results(
    cfg: DictConfig, pred, value, centroid_idx, caseid, decode_fn=None, q_ref=None
) -> Tuple[str, str, str]:
    pred_pressure = decode_fn(pred[0:1, :], 0, q_ref=q_ref).cpu().detach().numpy()
    pred_wallshearstress = decode_fn(pred[1:4, :], 1, q_ref=q_ref).cpu().detach().numpy()
    evals_results = {
        "cal_pressure_drag": pred_pressure,
        "cal_friction_resistance": pred_wallshearstress,
    }
    centroid = np.load(f"{cfg.reason_input_path}/centroid_{centroid_idx}.npy")

    centroid = centroid[:: cfg.subsample_eval, ...]

    cells = [("vertex", np.arange(tuple(centroid.shape)[0]).reshape(-1, 1))]

    os.makedirs(os.path.join(cfg.reason_output_path, "csv_vtp", str(int(value))), exist_ok=True)

    pred_pressure_csv_path = None
    pred_pressure_vtp_path = None
    pred_wallshearstress_csv_path = None
    pred_wallshearstress_vtp_path = None

    logging.info(evals_results.keys())
    for k, v in evals_results.items():
        array_hstack = np.hstack((centroid, v.T))
        csv_filename = os.path.join(
            cfg.reason_output_path,
            "csv_vtp",
            f"{int(value)}",
            f"{k}.csv",
        )
        np.savetxt(csv_filename, array_hstack, delimiter=",", fmt="%f")
        vtp_filename = os.path.join(
            cfg.reason_output_path,
            "csv_vtp",
            f"{int(value)}", 
            f"{k}.vtp",
        )
        if v.T.shape[1] == 1:
            save_vtp_from_dict(
                vtp_filename,
                {
                    "x": centroid[:, 0:1],
                    "y": centroid[:, 1:2],
                    "z": centroid[:, 2:3],
                    k: v.T,
                },
                ("x", "y", "z"),
                (k,),
            )
        else:
            save_vtp_from_dict(
                vtp_filename,
                {
                    "x": centroid[:, 0:1],
                    "y": centroid[:, 1:2],
                    "z": centroid[:, 2:3],
                    k: np.linalg.norm(v.T, axis=1, keepdims=True),
                },
                ("x", "y", "z"),
                (k,),
            )

        if k == "pred_pressure":
            pred_pressure_csv_path = csv_filename
            pred_pressure_vtp_path = vtp_filename
        elif k == "pred_wallshearstress":
            pred_wallshearstress_csv_path = csv_filename
            pred_wallshearstress_vtp_path = vtp_filename

    return (
        pred_pressure_csv_path,
        pred_pressure_vtp_path,
        pred_wallshearstress_csv_path,
        pred_wallshearstress_vtp_path,
    )


def quote_non_ascii_overrides(argv: List[str]) -> List[str]:
    """给含非 ASCII 字符（如中文路径）的 Hydra 覆盖参数值自动加引号。

    Hydra 的命令行 override 语法只允许未加引号的值使用 ASCII 字符，
    因此像 state=/path/模型训练/x.pdparams 这样的中文路径会触发
    LexerNoViableAltException。将值用双引号包裹后，Hydra 会把它当作
    普通字符串处理，从而支持中文等非 ASCII 字符。
    """
    result = []
    for arg in argv:
        # 只处理 key=value 形式的覆盖参数，跳过 --multirun 等选项。
        if arg.startswith("-") or "=" not in arg:
            result.append(arg)
            continue
        key, sep, value = arg.partition("=")
        already_quoted = (
            len(value) >= 2 and value[0] == value[-1] and value[0] in "'\""
        )
        if value and not already_quoted and not value.isascii():
            value = f'"{value}"'
        result.append(key + sep + value)
    return result


@hydra.main(version_base=None, config_path="./configs", config_name="inference")
def main(cfg: DictConfig):
    inference(cfg)


if __name__ == "__main__":
    sys.argv = quote_non_ascii_overrides(sys.argv)
    main()
