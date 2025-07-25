# pcb_env_sequential.py

import json
import numpy as np
from copy import deepcopy

def _safe_eval(x):
    return eval(x) if isinstance(x, str) else x

def transform_shape(pts, c_old, c_new, dθ):
    arr = np.array(pts, float)
    arr -= c_old
    θ = np.deg2rad(dθ)
    R = np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ),  np.cos(θ)]])
    arr = arr @ R.T
    arr += c_new
    return arr.astype(int).tolist()

class PCBRLEnv:
    """
    顺序放置环境：每一步只对下一个器件执行 (dx, dy, rot_idx) 动作，
    器件依 device_order（非主芯片按索引升序）顺序遍历。
    state 返回 {'layout': array(n_cells×3), 'current_device': idx}
    reward 只有最后一步才计算: reward = 1/(1+总布线长度)，否则为 0。
    """
    def __init__(self, pcb_json_path):
        with open(pcb_json_path, 'r', encoding='utf-8') as f:
            self.raw = json.load(f)
        self.grid_size = 256
        self.reset()

    def reset(self):
        d = deepcopy(self.raw)
        self.cells = d['cellList']
        self.nets  = d['netList']
        # 找主芯片（面积最大）
        areas = []
        for cell in self.cells:
            pts = np.array(_safe_eval(cell['contour']), int)
            xs, ys = pts[:,0], pts[:,1]
            areas.append((xs.max()-xs.min()) * (ys.max()-ys.min()))
        self.main_idx = int(np.argmax(areas))
        # 非主芯片索引，按升序
        self.device_order = [i for i in range(len(self.cells)) if i != self.main_idx]
        self.device_order.sort()
        self.n_dev = len(self.device_order)
        self.step_idx = 0
        self._refresh_cache()
        return self._get_state()

    def _refresh_cache(self):
        self.pos = []
        self.rot = []
        for cell in self.cells:
            self.pos.append(np.array(_safe_eval(cell['center']), int))
            self.rot.append(int(cell.get('rotation','0')))

    def _get_state(self):
        layout = np.stack([
            np.concatenate([self.pos[i], [self.rot[i]]])
            for i in range(len(self.cells))
        ])
        cur = self.device_order[self.step_idx] if self.step_idx < self.n_dev else -1
        return {'layout': layout, 'current_device': cur}

    def get_action_mask(self, step=32):
        """仅为当前 device 生成 (dx,dy,rot_idx) 合法动作列表"""
        if self.step_idx >= self.n_dev:
            return []
        idx = self.device_order[self.step_idx]
        main_c = self.pos[self.main_idx]
        mask = []
        rng = np.arange(-self.grid_size, self.grid_size, step, int)
        for rot_idx in range(4):
            for dx in rng:
                for dy in rng:
                    if self._is_valid(idx, main_c + np.array([dx,dy]), rot_idx):
                        mask.append((int(dx), int(dy), rot_idx))
        return mask

    def _is_valid(self, idx, new_c, rot_idx):
        """检查将 cell idx 放到 new_c 并旋转 rot_idx*90° 是否越界或重叠"""
        old_c = self.pos[idx]
        old_r = self.rot[idx]
        pts0  = _safe_eval(self.cells[idx]['contour'])
        new_pts = transform_shape(pts0, old_c, new_c, rot_idx*90 - old_r)
        arr = np.array(new_pts, int)
        # 越界
        if (arr < 0).any() or (arr[:,0] >= self.grid_size).any() or (arr[:,1] >= self.grid_size).any():
            return False
        # AABB 重叠检测
        x0,x1 = arr[:,0].min(), arr[:,0].max()
        y0,y1 = arr[:,1].min(), arr[:,1].max()
        for j, cell in enumerate(self.cells):
            if j == idx: continue
            op = np.array(_safe_eval(cell['contour']), int)
            ox0,ox1 = op[:,0].min(), op[:,0].max()
            oy0,oy1 = op[:,1].min(), op[:,1].max()
            if not (x1 < ox0 or ox1 < x0 or y1 < oy0 or oy1 < y0):
                return False
        return True

    def step(self, action):
        """
        action = (dx, dy, rot_idx)
        返回 state, reward, done, info
        """
        info = {}
        if self.step_idx >= self.n_dev:
            return self._get_state(), 0.0, True, info

        dx, dy, rot_idx = action
        idx = self.device_order[self.step_idx]
        main_c = self.pos[self.main_idx]
        new_c  = np.clip(main_c + np.array([dx,dy]), 0, self.grid_size-1).astype(int)

        # 非法动作
        if not self._is_valid(idx, new_c, rot_idx):
            return self._get_state(), 0.0, False, {'invalid': True}

        # 应用到 cell
        cell = self.cells[idx]
        old_c = self.pos[idx]; old_r = self.rot[idx]
        cell['contour'] = str(transform_shape(
            _safe_eval(cell['contour']), old_c, new_c, rot_idx*90 - old_r))
        # 更新 pins
        for pin in cell['pinList']:
            if 'center' in pin:
                p0 = _safe_eval(pin['center'])
            else:
                cp = np.array(_safe_eval(pin['contour']), int)
                p0 = cp.mean(axis=0).astype(int).tolist()
            new_p = transform_shape([p0], old_c, new_c, rot_idx*90 - old_r)[0]
            pin['center']  = str(new_p)
            pin['contour'] = str(transform_shape(
                _safe_eval(pin['contour']), old_c, new_c, rot_idx*90 - old_r))

        cell['center']   = str(new_c.tolist())
        cell['rotation'] = str(rot_idx*90)
        # 刷新缓存并移动到下一个 device
        self._refresh_cache()
        self.step_idx += 1

        # 如果完成所有 device，则计算 reward
        if self.step_idx >= self.n_dev:
            total = self._compute_wire_len()
            return self._get_state(), 1.0/(1.0+total), True, info

        # 否则每步 reward=0
        return self._get_state(), 0.0, False, info

    def _lookup_pin_coord(self, pin):
        """辅助：根据 pin['cellName'], pin['pinName'] 返回其实际 center"""
        for cell in self.cells:
            if cell['cellName'] == pin['cellName']:
                for p in cell['pinList']:
                    if p['pinName'] == pin['pinName']:
                        if 'center' in p:
                            return np.array(_safe_eval(p['center']), int)
                        pts = np.array(_safe_eval(p['contour']), int)
                        return pts.mean(axis=0).astype(int)
        return np.zeros(2, int)

    def _compute_wire_len(self):
        """用 A* 绕行两层计算真实总布线长度"""
        # 构建障碍网格
        g0 = np.zeros((self.grid_size, self.grid_size), bool)
        g1 = g0.copy()
        for cell in self.cells:
            pts = np.array(_safe_eval(cell['contour']), int)
            x0,x1 = pts[:,0].min(), pts[:,0].max()
            y0,y1 = pts[:,1].min(), pts[:,1].max()
            g0[y0:y1+1, x0:x1+1] = True
            g1[y0:y1+1, x0:x1+1] = True

        import heapq
        def astar(sx,sy,sl, tx,ty):
            dist = {(sx,sy,sl):0}
            pq   = [(abs(sx-tx)+abs(sy-ty), sx,sy,sl)]
            while pq:
                f,x,y,l = heapq.heappop(pq)
                g = dist[(x,y,l)]
                if (x,y,l)==(tx,ty,sl):
                    return g
                # 4 邻域
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx,ny,nl = x+dx, y+dy, l
                    if not (0<=nx<self.grid_size and 0<=ny<self.grid_size): continue
                    if (g0 if nl==0 else g1)[ny,nx]: continue
                    ng = g+1
                    if ng < dist.get((nx,ny,nl), 1e9):
                        dist[(nx,ny,nl)] = ng
                        h = abs(nx-tx)+abs(ny-ty)
                        heapq.heappush(pq,(ng+h,nx,ny,nl))
                # via 切层
                ol = 1-l
                if not (g0 if ol==0 else g1)[y,x]:
                    ng = g+1
                    if ng < dist.get((x,y,ol), 1e9):
                        dist[(x,y,ol)] = ng
                        h = abs(x-tx)+abs(y-ty)
                        heapq.heappush(pq,(ng+h,x,y,ol))
            return 1e5

        total = 0.0
        for net in self.nets:
            pins = net['pinList']
            for i in range(len(pins)-1):
                c1 = self._lookup_pin_coord(pins[i])
                c2 = self._lookup_pin_coord(pins[i+1])
                total += astar(c1[0],c1[1],0, c2[0],c2[1])
        return total
