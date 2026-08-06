#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from contextlib import asynccontextmanager
from timeit import default_timer
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import anyio
import hydra
import meshio
import numpy as np
import paddle
import math
import pyvista as pv
import vtk
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status
from omegaconf import DictConfig
from paddle.distributed import ParallelEnv
from paddle.distributed import fleet
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from pydantic import BaseModel

from ppcfd.models.ppfno.data import instantiate_inferencedatamodule
from ppcfd.models.ppfno.data.datamodule_lazy import compute_q_ref
from ppcfd.models.ppfno.losses import LpLoss
from ppcfd.models.ppfno.networks import instantiate_network
from ppcfd.models.ppfno.optim.schedulers import instantiate_scheduler
from ppcfd.models.ppfno.utils.average_meter import AverageMeter
from ppcfd.models.ppfno.utils.average_meter import AverageMeterDict
from ppcfd.models.ppfno.utils.dot_dict import DotDict
from ppcfd.models.ppfno.utils.dot_dict import flatten_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class InputData(BaseModel):
    pre_output_path: str
    reason_input_path: str
    reason_output_path: str
    save_eval_results: bool = False


class OutputData(BaseModel):
    error_code: int
    error_message: str
    cost_all: float
    cost_forward: float
    F_pred_modify: List[float]
    M_pred_modify: List[float]
    pred_pressure_csv_path: Union[str, None]
    pred_pressure_vtp_path: Union[str, None]
    pred_wallshearstress_csv_path: Union[str, None]
    pred_wallshearstress_vtp_path: Union[str, None]


# 模型
MODEL: paddle.nn.Layer = None
CFG = None


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


# 模型加载函数
def load_model():
    global MODEL
    try:
        MODEL = instantiate_network(CFG)
        if isinstance(MODEL, paddle.DataParallel):
            MODEL = MODEL._layers
        MODEL.eval()

        assert CFG.pd_path is not None, "checkpoint must be given."

        state = paddle.load(path=str(CFG.pd_path))
        MODEL.set_state_dict(state_dict=state["model"])
        device = ParallelEnv().device_id
        memory_allocated = paddle.device.cuda.memory_allocated(device=device) / (
            1024 * 1024 * 1024
        )
        logging.info(f"Memory usage with model loading: {memory_allocated:.2f} GB")

        logging.info(f"Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise


semaphore = None


# 应用生命周期事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    :param app:
    :return:
    """
    global semaphore
    # TODO() 设置并发数，例如2
    semaphore = asyncio.Semaphore(5)  # 在FastAPI启动时初始化
    # TODO() 启动时加载模型
    load_model()
    yield
    # 关闭时清理资源
    if MODEL is not None:
        # MODEL = None  # 或实际模型的清理代码
        pass


app = FastAPI(lifespan=lifespan)


# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}


async def async_save_eval_results(
    cfg, pred, value, indices, caseid, decode_fn, output: OutputData, q_ref=None
):
    try:
        (
            pred_pressure_csv_path,
            pred_pressure_vtp_path,
            pred_wallshearstress_csv_path,
            pred_wallshearstress_vtp_path,
        ) = save_eval_results(
            cfg,
            pred,
            value,
            indices[0],
            caseid,
            decode_fn=decode_fn,
            q_ref=q_ref,
        )
        
        # 更新输出对象中的文件路径
        output.pred_pressure_csv_path = pred_pressure_csv_path
        output.pred_pressure_vtp_path = pred_pressure_vtp_path
        output.pred_wallshearstress_csv_path = pred_wallshearstress_csv_path
        output.pred_wallshearstress_vtp_path = pred_wallshearstress_vtp_path
        
    except Exception as e:
        logging.error(f"Error in async file saving: {str(e)}")


# @app.post("/api/v1/inference", response_model=OutputData)
async def infer_model_task(input_data: InputData) -> OutputData:
    print(f"got input: {input_data}")
    global CFG
    CFG.pre_output_path = input_data.pre_output_path
    CFG.reason_input_path = input_data.reason_input_path
    CFG.reason_output_path = input_data.reason_output_path
    os.makedirs(os.path.join(CFG.reason_output_path, "log"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(CFG.reason_output_path, "log", "reason.log"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: %(message)s",
        force=True,
    )
    logging.info(f"开始处理请求(线程ID: {id(asyncio.get_running_loop())})")
    # await asyncio.sleep(5)
    with paddle.no_grad():
        global MODEL
        if MODEL is None:
            logging.error("Model not loaded")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        try:
            datamodule = instantiate_inferencedatamodule(
                CFG, CFG.reason_input_path, CFG.pre_output_path, CFG.n_inference_num
            )
            inference_dataloader = datamodule.inference_dataloader(
                enable_ddp=CFG.enable_ddp, batch_size=CFG.batch_size
            )
            all_files = os.listdir(CFG.reason_input_path)
            prefix = "area"
            indices = [item[5:9] for item in all_files if item.startswith(prefix)]

            def extract_number(s):
                return int(s)

            # create output json
            def create_json(json_file_path):
                if not os.path.exists(os.path.dirname(json_file_path)):
                    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

                if os.path.isfile(json_file_path):
                    os.remove(json_file_path)

                with open(json_file_path, "w") as file:
                    json.dump([], file)

            inference_json_file_path = os.path.join(
                CFG.reason_output_path,
                "json",
                "reason.json",
            )
            create_json(inference_json_file_path)

            def append_dict_to_json_list(file_path, dict_element):
                with open(file_path, "r") as file:
                    data = json.load(file)

                if isinstance(data, list):
                    data.append(dict_element)
                else:
                    logging.info("Error: The root of the JSON file is not a list.")
                    return
                with open(file_path, "w") as file:
                    json.dump(data, file, indent=4)

            indices.sort(key=extract_number)
            logging.info(f"Start evaluting {CFG.model} ...")
            eval_meter = AverageMeterDict()
            visualize_data_dicts = []
            loss_fn = LpLoss(size_average=True)

            def cal_mre(pred, label):
                return paddle.abs(x=pred - label) / paddle.abs(x=label)

            # for i, data_dict in enumerate(inference_dataloader):
            data_dict = next(iter(inference_dataloader))
            if ',' in data_dict['info'][0]['wind_speed']:
                value_list = [float(wind_speed) for wind_speed in data_dict['info'][0]['wind_speed'].split(',')]
                value_type = 'wind_speed'
            else:
                value_list = [float(wind_angle) for wind_angle in data_dict['info'][0]['wind_angle'].split(',')]
                value_type = 'wind_angle'
            
            for value in value_list:
                msg = ""
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
                    out_dict, pred, F_M_dict = MODEL.inference_dict(
                        device, data_dict, loss_fn=loss_fn, decode_fn=datamodule.decode
                    )
                    t2 = default_timer()
                    paddle.device.cuda.empty_cache()
                    msg += f"Inference (pure) took {t2 - t1:.2f} seconds."
                    pred_pressure_csv_path = None
                    pred_pressure_vtp_path = None
                    pred_wallshearstress_csv_path = None
                    pred_wallshearstress_vtp_path = None
                    if input_data.save_eval_results:
                        (
                            pred_pressure_csv_path,
                            pred_pressure_vtp_path,
                            pred_wallshearstress_csv_path,
                            pred_wallshearstress_vtp_path,
                        ) = get_pathes(
                            CFG,
                            value,
                            datamodule.inference_full_caseids[0],
                        )

                    output = OutputData(
                        error_code=0,
                        error_message="",
                        cost_forward=t2 - t1,
                        cost_all=0.0,
                        F_pred_modify = (F_M_dict["F_pred_modify"] / F_M_dict["F_const"]).cpu().numpy().tolist(),
                        M_pred_modify = (F_M_dict["M_pred_modify"] / F_M_dict["M_const"]).cpu().numpy().tolist(),
                        pred_pressure_csv_path=pred_pressure_csv_path,
                        pred_pressure_vtp_path=pred_pressure_vtp_path,
                        pred_wallshearstress_csv_path=pred_wallshearstress_csv_path,
                        pred_wallshearstress_vtp_path=pred_wallshearstress_vtp_path,
                    )
                    if input_data.save_eval_results:
                        
                        await async_save_eval_results(
                            CFG,
                            pred,
                            value,
                            indices,
                            datamodule.inference_full_caseids[0],
                            datamodule.decode,
                            output,
                            q_ref=data_dict["q_ref"][0],
                        )
                        

                except MemoryError as e:
                    logging.info(e)
                    if "Out of memory" in str(e):
                        logging.info(f"WARNING: OOM on sample {0}, skipping this sample.")
                        if hasattr(paddle.device.cuda, "empty_cache"):
                            paddle.device.cuda.empty_cache()
                        raise
                    else:
                        raise

                msg += f"Eval sample {0}... L2_Error: "
                for k, v in out_dict.items():
                    if k.split("_")[0] == "L2":
                        msg += f"{k}: {v.item():.4f}, "
                        eval_meter.update({k: v})
                msg += f"|| MRE and Value: "

                # 真·系数空间：折回物理力/力矩用于日志与统计
                F_pred_modify = F_M_dict["F_pred_modify"] / F_M_dict["F_const"]
                M_pred_modify = F_M_dict["M_pred_modify"] / F_M_dict["M_const"]
                F_pred = out_dict["F_pred"]
                M_pred = out_dict["M_pred"]
                eval_meter.update({"F_pred_modify": F_pred_modify})
                eval_meter.update({"M_pred_modify": M_pred_modify})
                msg += f"F_pred_modify: {F_pred_modify.numpy()}, "
                msg += f"M_pred_modify: {M_pred_modify.numpy()}, "

                inference_json_dict["parts"] = os.path.basename(CFG.reason_input_path)

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

                    # 真·系数空间：modify 分支已是无量纲系数，cal_value 需 ×(1/const) 还原物理量
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

            msg = f"Inference took {t3 - t1:.2f} seconds. Everage eval values: "
            eval_dict = eval_meter.avg
            for k, v in eval_dict.items():
                msg += f"{v}({k}), "
            logging.info(msg)
            max_memory_allocated = paddle.device.cuda.max_memory_allocated(
                device=device
            ) / (1024 * 1024 * 1024)
            logging.info(f"Memory Usage: {max_memory_allocated:.2f} GB (MAX).")

            logging.info(f"请求处理完成(线程ID: {id(asyncio.get_running_loop())})")
            print('output',output)
            return output
        except Exception as e:
            logging.error(f"请求处理出现错误(线程ID: {id(asyncio.get_running_loop())})")
            logging.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}",
            )


@app.post("/api/v1/inference", response_model=OutputData)
async def infer_model(input_data: InputData) -> OutputData:
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    try:
        async with semaphore:
            # TODO() 这里设置单个请求的超时时间，业务层超时
            result = await asyncio.wait_for(infer_model_task(input_data), timeout=60)
            print('result',result)
            return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="request timeout > 60s",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


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


def get_pathes(
    cfg: DictConfig, value, caseid
) -> Tuple[str, str, str, str]:
    evals_results = {
        "pred_pressure": None,
        "pred_wallshearstress": None,
    }

    pred_pressure_csv_path = None
    pred_pressure_vtp_path = None
    pred_wallshearstress_csv_path = None
    pred_wallshearstress_vtp_path = None

    for k, v in evals_results.items():
        csv_filename = os.path.join(
            cfg.reason_output_path,
            "csv_vtp",
            f"{int(value)}",
            f"{k}.csv",
        )
        vtp_filename = os.path.join(
            cfg.reason_output_path,
            "csv_vtp",
            f"{int(value)}", 
            f"{k}.vtp",
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
    因此像 pd_path=/path/模型训练/x.pdparams 这样的中文路径会触发
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
    global CFG
    CFG = cfg
    from omegaconf import OmegaConf
    import logging
    logging.warning(OmegaConf.to_yaml(CFG))
    import uvicorn

    port = os.getenv("main", "8087")
    print('port',port)
    uvicorn.run(app, host="0.0.0.0", workers=1, port=int(port))


if __name__ == "__main__":
    sys.argv = quote_non_ascii_overrides(sys.argv)
    main()
