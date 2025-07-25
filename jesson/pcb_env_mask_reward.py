# pcb_env_mask_reward.py

import json
import numpy as np
import heapq
from copy import deepcopy

def _safe_eval(x):
    return eval(x) if isinstance(x, str) else x

def transform_shape(shape_pts, center_old, center_new, rotation_deg):
    pts = np.array(shape_pts, dtype=float)
    pts -= center_old
    θ = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ),  np.cos(θ)]])
    pts = pts @ R.T
    pts += center_new
    return pts.astype(int).tolist()

class PCBRLEnv:
    def __init__(self, pcb_json_path, max_steps=1000):
        with open(pcb_json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        self.grid_size = 256
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.data      = deepcopy(self.raw_data)
        self.cell_list = self.data['cellList']
        self.net_list  = self.data['netList']
        self.main_idx  = self._find_main_idx()
        self.n_cells   = len(self.cell_list)
        self.non_main  = [i for i in range(self.n_cells) if i != self.main_idx]
        self._refresh_cache()
        self.placed_devices = set()
        self.step_count     = 0
        return self._get_state()

    def _find_main_idx(self):
        areas = []
        for cell in self.raw_data['cellList']:
            pts = np.array(_safe_eval(cell['contour']),dtype=int)
            xs, ys = pts[:,0], pts[:,1]
            areas.append((xs.max()-xs.min())*(ys.max()-ys.min()))
        return int(np.argmax(areas))

    def _refresh_cache(self):
        self.cell_pos = []
        self.cell_rot = []
        for cell in self.cell_list:
            c = np.array(_safe_eval(cell['center']),dtype=int)
            r = int(cell.get('rotation','0'))
            self.cell_pos.append(c)
            self.cell_rot.append(r)

    def _get_state(self):
        return np.stack([
            np.concatenate([self.cell_pos[i],[self.cell_rot[i]]])
            for i in range(self.n_cells)
        ])

    def is_valid_action(self, action):
        dev_choice, rel_x, rel_y, rot_idx = action
        cell_idx = self.non_main[dev_choice]
        main_c   = self.cell_pos[self.main_idx]
        new_c    = np.array([main_c[0]+rel_x, main_c[1]+rel_y],dtype=int)

        cell = self.cell_list[cell_idx]
        old_c = np.array(_safe_eval(cell['center']),dtype=int)
        old_r = self.cell_rot[cell_idx]
        contour_pts = _safe_eval(cell['contour'])
        new_contour = transform_shape(contour_pts, old_c, new_c, rot_idx*90 - old_r)
        pts = np.array(new_contour,dtype=int)

        # 越界
        if (pts<0).any() or (pts[:,0]>=self.grid_size).any() or (pts[:,1]>=self.grid_size).any():
            return False

        # AABB 重叠
        x0,x1 = pts[:,0].min(), pts[:,0].max()
        y0,y1 = pts[:,1].min(), pts[:,1].max()
        for j, other in enumerate(self.cell_list):
            if j==cell_idx: continue
            opts = np.array(_safe_eval(other['contour']),dtype=int)
            ox0,ox1 = opts[:,0].min(), opts[:,0].max()
            oy0,oy1 = opts[:,1].min(), opts[:,1].max()
            if not (x1<ox0 or ox1<x0 or y1<oy0 or oy1<y0):
                return False
        return True

    def get_action_mask(self, dev_choice, step=16):
        mask = []
        main_c = self.cell_pos[self.main_idx]
        rng    = np.arange(-self.grid_size,self.grid_size,step,dtype=int)
        for rot_idx in range(4):
            for dx in rng:
                for dy in rng:
                    act = (dev_choice,int(dx),int(dy),rot_idx)
                    if self.is_valid_action(act):
                        mask.append(act)
        return mask

    def step(self, action):
        dev_choice, rel_x, rel_y, rot_idx = action
        info = {'invalid':False}

        if not self.is_valid_action(action):
            info['invalid'] = True
            return self._get_state(), 0.0, False, info

        cell_idx = self.non_main[dev_choice]
        self.placed_devices.add(dev_choice)

        main_c = self.cell_pos[self.main_idx]
        new_c  = np.clip([main_c[0]+rel_x, main_c[1]+rel_y],0,self.grid_size-1).astype(int)
        new_r  = (rot_idx*90)%360

        cell = self.cell_list[cell_idx]
        old_c = np.array(_safe_eval(cell['center']),dtype=int)
        old_r = int(cell.get('rotation','0'))

        # 更新 contour
        pts0 = _safe_eval(cell['contour'])
        cell['contour'] = str(transform_shape(pts0, old_c, new_c, new_r-old_r))
        # 更新 pin
        for pin in cell['pinList']:
            if 'center' in pin:
                p0 = _safe_eval(pin['center'])
            else:
                cp = np.array(_safe_eval(pin['contour']),dtype=int)
                p0 = cp.mean(axis=0).astype(int).tolist()
            new_p0 = transform_shape([p0], old_c, new_c, new_r-old_r)[0]
            pin['center'] = str(new_p0)
            pin['contour'] = str(transform_shape(_safe_eval(pin['contour']), old_c, new_c, new_r-old_r))

        cell['center']   = str(new_c.tolist())
        cell['rotation'] = str(new_r)

        self._refresh_cache()
        self.step_count += 1

        # 未放完
        if len(self.placed_devices) < len(self.non_main):
            return self._get_state(), 0.0, False, info

        # 全部放完：正 reward
        total_len = self._compute_wire_length()
        reward    = 1.0/(1.0+total_len)
        return self._get_state(), reward, True, info

    def _compute_wire_length(self):
        """
        使用简化的 A* 或曼哈顿距离计算路由长度。
        从 cellList 找到 pin 的实际坐标，不再直接用 netList 中的 pin dict。
        """
        # 构建两层障碍网格（与之前保持一致）
        g0 = np.zeros((self.grid_size, self.grid_size), bool)
        g1 = g0.copy()
        for cell in self.cell_list:
            pts = np.array(_safe_eval(cell['contour']), dtype=int)
            x0, x1 = pts[:, 0].min(), pts[:, 0].max()
            y0, y1 = pts[:, 1].min(), pts[:, 1].max()
            g0[y0:y1 + 1, x0:x1 + 1] = True
            g1[y0:y1 + 1, x0:x1 + 1] = True

        # 简化的曼哈顿距离代替 A*
        def route_len(c1, c2):
            return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

        total = 0.0

        for net in self.net_list:
            pins = net['pinList']
            # 对 net 中相邻两 pin 计算距离
            for i in range(len(pins) - 1):
                p1 = pins[i]
                p2 = pins[i + 1]

                # 从 cellList 查找 pin 对象
                def find_pin_coord(cell_name, pin_name):
                    for cell in self.cell_list:
                        if cell['cellName'] == cell_name:
                            for pin in cell['pinList']:
                                if pin['pinName'] == pin_name:
                                    # 优先用 pin['center']
                                    if 'center' in pin:
                                        return np.array(_safe_eval(pin['center']), dtype=int)
                                    # 否则用 contour 的几何中心
                                    cpts = np.array(_safe_eval(pin['contour']), dtype=int)
                                    return cpts.mean(axis=0).astype(int)
                    # 万一没找到，返 (0,0)
                    return np.zeros(2, dtype=int)

                c1 = find_pin_coord(p1['cellName'], p1['pinName'])
                c2 = find_pin_coord(p2['cellName'], p2['pinName'])
                total += route_len(c1, c2)

        return total

