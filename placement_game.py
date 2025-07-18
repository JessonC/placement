import numpy as np
import json
import os
from copy import deepcopy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_layout(cell_list, centers, show=True):
    fig, ax = plt.subplots(figsize=(8,8))
    for cell, center in zip(cell_list, centers):
        # 解析轮廓，考虑旋转
        contour = eval(cell["contour"])
        # 如果有rotation则需要对轮廓做旋转
        rot = int(cell.get("rotation", "0"))
        center_np = np.array(center)
        contour_np = np.array(contour) - np.array(eval(cell["center"]))
        if rot != 0:
            angle = np.deg2rad(rot)
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle),  np.cos(angle)]])
            contour_np = contour_np @ rot_matrix.T
        contour_real = contour_np + center_np
        xs = list(contour_real[:,0]) + [contour_real[0,0]]
        ys = list(contour_real[:,1]) + [contour_real[0,1]]
        ax.plot(xs, ys, linewidth=2, label=cell["cellName"])
        # 画中心点
        ax.plot(center_np[0], center_np[1], 'ro')
        ax.text(center_np[0], center_np[1], cell["cellName"], color='r', fontsize=8, ha='center', va='center')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_aspect('equal')
    ax.set_title("PCB layout")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()

import json
import numpy as np
from copy import deepcopy

def transform_shape(shape, center_old, center_new, rotation_deg):
    """
    shape: list of [x, y]点，原始坐标（以center_old为参考）
    center_old: [x, y]，原始中心
    center_new: [x, y]，新的中心
    rotation_deg: 旋转角（度）
    返回变换后的新shape
    """
    pts = np.array(shape)
    angle = np.deg2rad(rotation_deg)
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    pts = pts - center_old   # 平移到原点
    pts = pts @ rot_mat.T    # 旋转
    pts = pts + center_new   # 平移到新中心
    return pts.astype(int).tolist()


class PCBRLEnv:
    def __init__(self, pcb_json_path):
        with open(pcb_json_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        self.grid_size = 256
        self.n_quadrant = 16
        self.n_fine_move = 5
        self.n_fine_step = 5
        self.n_rotate = 4
        self.reset()

    def reset(self):
        self.data = deepcopy(self.raw_data)
        self.cell_list = self.data["cellList"]
        self.net_list = self.data["netList"]
        self.main_idx = self._find_main_idx()
        self.n_cells = len(self.cell_list)
        self.actions = []
        self._refresh_cache()
        return self._get_state()

    def _find_main_idx(self):
        areas = []
        for cell in self.cell_list:
            contour = np.array(eval(cell["contour"]))
            xs, ys = contour[:, 0], contour[:, 1]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            areas.append(area)
        return np.argmax(areas)

    def _refresh_cache(self):
        self.cell_pos = []
        self.cell_rot = []
        for cell in self.cell_list:
            center = np.array(eval(cell["center"]))
            rot = int(cell.get("rotation", "0"))
            self.cell_pos.append(center)
            self.cell_rot.append(rot)

    def _get_state(self):
        state = []
        for i in range(self.n_cells):
            state.append(np.concatenate([self.cell_pos[i], [self.cell_rot[i]]]))
        return np.stack(state)

    def get_cell_names(self):
        return [cell["cellName"] for cell in self.cell_list]

    def step(self, actions):
        assert len(actions) == self.n_cells - 1, "必须对每个非主芯片都出决策"
        for i, act in enumerate(actions):
            cell_idx = i if i < self.main_idx else i + 1
            self._apply_action(cell_idx, act)
        self._refresh_cache()
        reward = self._compute_reward()
        done = True
        return self._get_state(), reward, done, {}

    def _apply_action(self, idx, action):
        cell = self.cell_list[idx]
        center = self.cell_pos[idx]
        rot = self.cell_rot[idx]

        # 记录action前的center、rot
        center_old = np.array(eval(cell["center"]))
        rot_old = int(cell.get("rotation", "0"))

        quadrant_type, quadrant_idx, quadrant_dist, fine_dir, fine_step, rot_mode = action

        # ======= 动作执行部分 =======
        if quadrant_type != 0:
            main_center = self.cell_pos[self.main_idx]
            step = self.grid_size // 4
            row, col = divmod(quadrant_idx, 4)
            quad_x = int(main_center[0] + (col - 1.5) * step)
            quad_y = int(main_center[1] + (row - 1.5) * step)
            dir_vec = main_center - np.array([quad_x, quad_y])
            dist = int(quadrant_dist * step / 4)
            new_center = np.array([quad_x, quad_y]) + dir_vec / (np.linalg.norm(dir_vec) + 1e-8) * dist
            center = new_center.astype(int)

        move_delta = np.zeros(2)
        if fine_dir == 1:
            move_delta[1] = -fine_step
        elif fine_dir == 2:
            move_delta[1] = fine_step
        elif fine_dir == 3:
            move_delta[0] = -fine_step
        elif fine_dir == 4:
            move_delta[0] = fine_step

        center = np.clip(center + move_delta, 0, self.grid_size - 1)

        if rot_mode == 1:
            rot = (rot + 90) % 360
        elif rot_mode == 2:
            rot = (rot - 90) % 360
        elif rot_mode == 3:
            rot = (rot + 180) % 360

        # ======= 信息同步部分 =======
        delta_rot = rot - rot_old
        center_new = center
        # 更新contour
        contour_pts = eval(cell["contour"])
        new_contour = transform_shape(contour_pts, center_old, center_new, delta_rot)
        cell["contour"] = str(new_contour)
        # 更新pin center 和 contour
        for pin in cell["pinList"]:
            pin_center_old = np.array(eval(pin["center"]))
            pin_contour_old = eval(pin["contour"])
            new_pin_center = transform_shape([pin_center_old], center_old, center_new, delta_rot)[0]
            new_pin_contour = transform_shape(pin_contour_old, center_old, center_new, delta_rot)
            pin["center"] = str(new_pin_center)
            pin["contour"] = str(new_pin_contour)

        # ======= 状态同步到env内部缓存 =======
        self.cell_pos[idx] = center
        self.cell_rot[idx] = rot
        cell["center"] = str(center.tolist())
        cell["rotation"] = str(rot)

    def _compute_reward(self):
        pin_coord_dict = {}
        for idx, cell in enumerate(self.cell_list):
            center, rot = self.cell_pos[idx], self.cell_rot[idx]
            angle = np.deg2rad(rot)
            mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            for pin in cell["pinList"]:
                pin_local = np.array(eval(pin["center"])) - np.array(eval(cell["center"]))
                pin_real = np.dot(mat, pin_local) + center
                pin_coord_dict[(cell["cellName"], pin["pinName"])] = pin_real

        total_dist = 0
        for net in self.net_list:
            pinList = net["pinList"]
            for i in range(len(pinList) - 1):
                for j in range(i + 1, len(pinList)):
                    p1 = pin_coord_dict[(pinList[i]["cellName"], pinList[i]["pinName"])]
                    p2 = pin_coord_dict[(pinList[j]["cellName"], pinList[j]["pinName"])]
                    total_dist += np.linalg.norm(p1 - p2)

        return -total_dist

# class PCBRLEnv:
#     def __init__(self, pcb_json_path):
#         with open(pcb_json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         self.raw_data = deepcopy(data)
#         self.grid_size = 256
#         self.reset()
#         # action space定义
#         self.n_quadrant = 16
#         self.n_fine_move = 5  # 不动、上、下、左、右
#         self.n_fine_step = 5  # 1~5格
#         self.n_rotate = 4     # 不转，顺时针90，逆时针90，180
#         # 编码方案：[粗调类型, 粗调象限, 粗调距离], [细调方向, 细调步数], [旋转方式]
#
#     def reset(self):
#         self.data = deepcopy(self.raw_data)
#         self.cell_list = self.data["cellList"]
#         self.net_list = self.data["netList"]
#         self.main_idx = self._find_main_idx()
#         self.n_cells = len(self.cell_list)
#         self.actions = []
#         self._refresh_cache()
#         return self._get_state()
#
#     def _find_main_idx(self):
#         # 默认主芯片为面积最大者
#         areas = []
#         for cell in self.raw_data["cellList"]:
#             contour = eval(cell["contour"])
#             xs = [p[0] for p in contour]
#             ys = [p[1] for p in contour]
#             areas.append((max(xs)-min(xs))*(max(ys)-min(ys)))
#         return np.argmax(areas)
#
#     def _refresh_cache(self):
#         # 更新所有cell的center, contour, pin的center/contour到np数组，旋转等
#         self.cell_pos = []
#         self.cell_rot = []
#         for cell in self.cell_list:
#             center = np.array(eval(cell["center"]))
#             self.cell_pos.append(center)
#             # rotation一定要是整数
#             self.cell_rot.append(int(cell.get("rotation", "0")))
#
#     def _get_state(self):
#         # 返回所有cell的(center, rotation)，可附加feature
#         state = []
#         for i, cell in enumerate(self.cell_list):
#             center = np.array(eval(cell["center"]))
#             rot = int(cell.get("rotation", "0"))
#             state.append(np.concatenate([center, [rot]]))
#         return np.stack(state)
#
#     def get_cell_names(self):
#         return [cell["cellName"] for cell in self.cell_list]
#
#     def step(self, actions):
#         # actions: n_cells-1个，每个形如(action1, action2, action3)
#         assert len(actions) == self.n_cells-1, "必须对每个非主芯片都出决策"
#         for i, act in enumerate(actions):
#             cell_idx = i if i < self.main_idx else i+1  # 跳过主芯片
#             self._apply_action(cell_idx, act)
#         self._refresh_cache()
#         reward = self._compute_reward()
#         done = True  # 一轮就done
#         return self._get_state(), reward, done, {}
#
#     def _apply_action(self, idx, action):
#         # action = (粗调类型, 粗调象限, 粗调距离, 细调方向, 细调步数, 旋转)
#         # idx: cell_list下标
#         cell = self.cell_list[idx]
#         center = np.array(eval(cell["center"]))
#         rot = int(cell.get("rotation", "0"))
#
#         # 1. 粗调
#         quadrant_type, quadrant_idx, quadrant_dist = action[0:3]
#         if quadrant_type != 0:
#             # 象限参考主芯片
#             main_center = np.array(eval(self.cell_list[self.main_idx]["center"]))
#             step = self.grid_size // 4  # 每象限边长
#             row = quadrant_idx // 4
#             col = quadrant_idx % 4
#             quad_x = int(main_center[0] + (col-1.5)*step)
#             quad_y = int(main_center[1] + (row-1.5)*step)
#             # 距离向主芯片中心靠拢（可自定义），或取象限中心到main中心的向量，再缩放
#             dir_vec = main_center - np.array([quad_x, quad_y])
#             dist = int(quadrant_dist*step/4)
#             new_center = np.array([quad_x, quad_y]) + dir_vec/np.linalg.norm(dir_vec+1e-8)*dist
#             center = new_center.astype(int)
#         # 2. 细调
#         fine_dir, fine_step = action[3:5]
#         move_delta = np.zeros(2)
#         if fine_dir == 1:   # 上
#             move_delta[1] = -fine_step
#         elif fine_dir == 2: # 下
#             move_delta[1] = fine_step
#         elif fine_dir == 3: # 左
#             move_delta[0] = -fine_step
#         elif fine_dir == 4: # 右
#             move_delta[0] = fine_step
#         center = (center + move_delta).clip(0, self.grid_size-1)
#
#         # 3. 旋转
#         rot_mode = action[5]
#         if rot_mode == 1:
#             rot = (rot + 90)%360
#         elif rot_mode == 2:
#             rot = (rot - 90)%360
#         elif rot_mode == 3:
#             rot = (rot + 180)%360
#         # 更新数据
#         cell["center"] = str(center.tolist())
#         cell["rotation"] = str(rot)
#         # contour和pin需要根据新的center/rotation同步重算（略，可用工具函数实现）
#
#     def _compute_reward(self):
#         # 全局pin-to-pin欧氏距离之和的负数
#         # 先统计所有pin全局坐标
#         pin_coord_dict = {}
#         for cell in self.cell_list:
#             center = np.array(eval(cell["center"]))
#             rot = int(cell.get("rotation", "0"))
#             for pin in cell["pinList"]:
#                 pin_local = np.array(eval(pin["center"])) - np.array(eval(cell["center"]))  # local offset
#                 # 旋转
#                 angle = np.deg2rad(rot)
#                 mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
#                 pin_real = np.dot(mat, pin_local) + center
#                 pin_coord_dict[(cell["cellName"], pin["pinName"])] = pin_real
#         # 统计所有net的pin距离
#         total_dist = 0
#         for net in self.net_list:
#             pinList = net["pinList"]
#             for i in range(len(pinList)-1):
#                 for j in range(i+1, len(pinList)):
#                     p1 = pin_coord_dict[(pinList[i]["cellName"], pinList[i]["pinName"])]
#                     p2 = pin_coord_dict[(pinList[j]["cellName"], pinList[j]["pinName"])]
#                     total_dist += np.linalg.norm(p1-p2)
#         return -total_dist  # 距离越短，reward越高

# --------------------- 人工操作接口 ---------------------
def manual_run_env(env: PCBRLEnv):
    state = env.reset()
    cell_names = env.get_cell_names()
    main_idx = env.main_idx
    # Ux按数字排序
    def key_ux(name):
        if name.startswith("U"):
            try:
                return int(name[1:])
            except:
                return 9999
        else:
            return 9999
    sorted_indices = sorted(
        [(i, name) for i, name in enumerate(cell_names) if i != main_idx],
        key=lambda x: key_ux(x[1])
    )
    print(f"main cell: {cell_names[main_idx]}")
    actions = []
    for i, name in sorted_indices:
        print(f"\n器件 {name} 调整:")
        # 1. 粗调
        q_type = int(input("粗调类型(0:不调, 1:调): "))
        if q_type == 0:
            q_idx, q_dist = 0, 0
        else:
            q_idx = int(input("粗调象限(0-15): "))
            q_dist = int(input("粗调距离(0-4): "))
        # 2. 细调
        f_dir = int(input("细调(0:不动 1:上 2:下 3:左 4:右): "))
        if f_dir == 0:
            f_step = 0
        else:
            f_step = int(input("细调步数(1-5): "))
        # 3. 旋转
        rot_mode = int(input("旋转(0:不转, 1:顺90, 2:逆90, 3:180): "))
        actions.append([q_type, q_idx, q_dist, f_dir, f_step, rot_mode])
        # 实时执行单个action并可视化
        env._apply_action(i, [q_type, q_idx, q_dist, f_dir, f_step, rot_mode])
        env._refresh_cache()
        # 可视化当前状态
        try:
            plot_layout(env.cell_list, [np.array(eval(cell["center"])) for cell in env.cell_list], show=True)
        except Exception as e:
            print("plot error:", e)
    # 最终reward统计
    env._refresh_cache()
    reward = env._compute_reward()
    print("Final Reward:", reward)



if __name__ == "__main__":
    pcb_path = os.path.join("pcb_pre_jsons", os.listdir("pcb_pre_jsons")[1])
    env = PCBRLEnv(pcb_path)
    manual_run_env(env)
