import json as js
import numpy as np
from pathlib import Path



class AnnotationTransformer:
    def __init__(self, path_to_params: Path):
        self.path_to_params = path_to_params #"../simulation/simulation_params.json"
        with open(path_to_params, "r") as f:
            self.params = js.load(f)    


    def transform(self, r_rel: np.ndarray) -> tuple:
        # координаты опорной точки, относительно которой задаются координаты
        r0 = np.array(self.params["scene_model"]["groups"][0]["reference_point"])
        # абсолютные координаты опорной точки
        r = r0 + r_rel
        # начальное положение аппарата
        Rs0 = np.array(self.params["platform_models"][0]["trajectory_model"]["r0"])
        # скорость аппарата (постоянная)
        Vs = np.array(self.params["platform_models"][0]["trajectory_model"]["v"])
        # начальный момент времени для траектории аппарата
        t0 = np.array(self.params["platform_models"][0]["trajectory_model"]["t0"])
        # 
        dL = np.dot(r - Rs0, Vs/np.linalg.norm(Vs))
        #
        dt = dL / np.linalg.norm(Vs)
        # момент времени, в который частота Дополера равна нулю (то есть линия, соединяющая аппарат и цель перпендикулярна скорости аппарата)
        t = t0 + dt
        # координаты аппарата в момент времени t (прямолинейное движение)
        Rs = Rs0 + Vs*dt
        # расстояние между аппаратом и целью в момент времени t
        dist = np.linalg.norm(r - Rs)
        # скорость света
        c = 3e8
        
        # частота дискретизации АЦП
        fs = self.params["time_grids"]["system_time_grid"]
        # время начала окна приема относительно начала строба
        SWST = self.params["cyclograms"][1]["start_sample"]/fs
        # параметры циклограммы приема
        burst_params = self.params["cyclograms"][0]["bursts"][0]
        # период повторения окон
        pri = (burst_params["samples_per_window"] + burst_params["samples_between_windows"])/fs
        # частота повторения окон
        prf = 1.0 / pri
        # задержка сигнала
        delay = 2*dist/c
        # кол-во периодов в задержке
        rank = int(np.floor(delay*prf))
        # дробная часть задержке
        tau = delay - rank/prf
        # индекс строки
        lineI = int(np.round(t*prf)) - rank
        # интедк столбца
        colI = int(np.round((tau - SWST)*fs))
        #print(lineI, colI)
        return lineI, colI


    
    
    def transform_bbox(self, x_bbox_indexes, y_bbox_indexes, z_bbox_indexes):
        assert len(x_bbox_indexes) == len(y_bbox_indexes) == len(z_bbox_indexes) == 2
        left_point = np.array([y_bbox_indexes[0], x_bbox_indexes[0], z_bbox_indexes[0]])
        right_point = np.array([y_bbox_indexes[1], x_bbox_indexes[1], z_bbox_indexes[1]])
        line_left, col_left = self.transform(left_point)
        line_right, col_right = self.transform(right_point)

        return (line_left, col_left), (line_right, col_right)