# pcb_env_mask_reward.py

import json
import numpy as np
import heapq
from copy import deepcopy

def transform_shape(shape_pts, center_old, center_new, rotation_deg):
    """
    将一组点 shape_pts 从 center_old 平移、旋转 rotation_deg，再平移到 center_new。
    shape_pts: list of [x,y]
    center_old: [x,y]
    center_new: [x,y]
    rotation_deg: 旋转角度（度）
    """
    pts = np.array(shape_pts, dtype=float)
    # 平移到原点
    pts -= center_old
    # 旋转
    θ = np.deg2rad(rotation_deg)
    R = np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ),  np.cos(θ)]])
    pts = pts @ R.T
    # 平移到新中心
    pts += center_new
    return pts.astype(int).tolist()

class PCBRLEnv:
    """
    PCB 强化学习环境，动作定义为：
      (device_choice, rel_x, rel_y, rot_idx)
    并提供 action-mask 过滤非法放置（重叠/越界）。
    奖励使用两层 A* 路由长度的负和。
    """
    def __init__(self, pcb_json_path, max_steps=1000):
        # 读取 JSON
        with open(pcb_json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        self.grid_size = 256
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        """重置环境到初始布局，返回初始 state"""
        self.data      = deepcopy(self.raw_data)
        self.cell_list = self.data['cellList']
        self.net_list  = self.data['netList']
        # 主芯片为面积最大的
        self.main_idx = self._find_main_idx()
        self.n_cells  = len(self.cell_list)
        # 可动器件索引（排除主芯片）
        self.non_main = [i for i in range(self.n_cells) if i != self.main_idx]
        # 缓存坐标/旋转
        self._refresh_cache()
        self.step_count = 0
        return self._get_state()

    def _find_main_idx(self):
        areas = []
        for cell in self.raw_data['cellList']:
            pts = np.array(eval(cell['contour']), dtype=int)
            xs, ys = pts[:,0], pts[:,1]
            areas.append((xs.max()-xs.min())*(ys.max()-ys.min()))
        return int(np.argmax(areas))

    def _refresh_cache(self):
        """同步 self.cell_pos 和 self.cell_rot"""
        self.cell_pos = []
        self.cell_rot = []
        for cell in self.cell_list:
            c = np.array(eval(cell['center']), dtype=int)
            r = int(cell.get('rotation','0'))
            self.cell_pos.append(c)
            self.cell_rot.append(r)

    def _get_state(self):
        """返回 state: shape = (n_cells, 3), 每行 [x, y, rot]"""
        arr = np.stack([
            np.concatenate([self.cell_pos[i],[self.cell_rot[i]]])
            for i in range(self.n_cells)
        ])
        return arr

    def is_valid_action(self, action):
        """
        检查 action 是否合法（不重叠、不越界）。
        action = (device_choice, rel_x, rel_y, rot_idx)
        """
        dev_choice, rel_x, rel_y, rot_idx = action
        cell_idx = self.non_main[dev_choice]
        main_c   = self.cell_pos[self.main_idx]
        new_c    = np.array([main_c[0]+rel_x, main_c[1]+rel_y], dtype=int)

        # 更新 contour 后的包围矩形检查
        old_ctr = np.array(eval(self.cell_list[cell_idx]['center']),dtype=int)
        pts = np.array(transform_shape(
            eval(self.cell_list[cell_idx]['contour']),
            old_ctr, new_c, rot_idx*90 - self.cell_rot[cell_idx]
        ), dtype=int)
        # 越界检查
        if (pts < 0).any() or (pts[:,0]>=self.grid_size).any() or (pts[:,1]>=self.grid_size).any():
            return False

        # AABB 重叠检查
        x0, x1 = pts[:,0].min(), pts[:,0].max()
        y0, y1 = pts[:,1].min(), pts[:,1].max()
        for j, cell in enumerate(self.cell_list):
            if j == cell_idx: continue
            other = np.array(eval(cell['contour']), dtype=int)
            ox0, ox1 = other[:,0].min(), other[:,0].max()
            oy0, oy1 = other[:,1].min(), other[:,1].max()
            if not (x1 < ox0 or ox1 < x0 or y1 < oy0 or oy1 < y0):
                return False
        return True

    def get_action_mask(self, dev_choice, step=16):
        """
        返回该器件所有合法 action 列表：
          (dev_choice, rel_x, rel_y, rot_idx)
        step: 相对坐标离散步长，越小动作空间越大。
        """
        mask = []
        main_c = self.cell_pos[self.main_idx]
        rng = np.arange(-self.grid_size, self.grid_size, step, dtype=int)
        for rot_idx in range(4):
            for dx in rng:
                for dy in rng:
                    act = (dev_choice, dx, dy, rot_idx)
                    if self.is_valid_action(act):
                        mask.append(act)
        return mask

    def step(self, action):
        """
        执行 action，并返回 (state, reward, done, info)
        info['invalid']=True 表示非法 action。
        """
        dev_choice, rel_x, rel_y, rot_idx = action
        if not self.is_valid_action(action):
            # 非法 action 直接大惩罚
            return self._get_state(), -1e4, False, {'invalid':True}

        cell_idx = self.non_main[dev_choice]
        main_c   = self.cell_pos[self.main_idx]
        new_c    = np.clip(
            np.array([main_c[0]+rel_x, main_c[1]+rel_y]),
            0, self.grid_size-1
        ).astype(int)
        new_r    = (rot_idx * 90) % 360

        cell    = self.cell_list[cell_idx]
        old_c   = np.array(eval(cell['center']),dtype=int)
        old_r   = int(cell.get('rotation','0'))

        # 更新 contour
        cell['contour'] = str(transform_shape(
            eval(cell['contour']), old_c, new_c, new_r-old_r))
        # 更新 pin
        for pin in cell['pinList']:
            p0 = eval(pin['center'])
            c0 = eval(pin['contour'])
            new_p0 = transform_shape([p0], old_c, new_c, new_r-old_r)[0]
            new_c0 = transform_shape(c0,   old_c, new_c, new_r-old_r)
            pin['center']  = str(new_p0)
            pin['contour'] = str(new_c0)
        # 更新自身 center/rotation
        cell['center']   = str(new_c.tolist())
        cell['rotation'] = str(new_r)

        # 同步缓存
        self._refresh_cache()
        self.step_count += 1

        # 计算 reward
        reward = self._compute_reward()
        done   = (self.step_count >= self.max_steps)
        return self._get_state(), reward, done, {'invalid':False}

    def _compute_reward(self):
        """
        基于两层 A* 路由长度负和作为 reward。
        """
        # 构造两层障碍网格
        g0 = np.zeros((self.grid_size,self.grid_size),bool)
        g1 = np.zeros_like(g0)
        for cell in self.cell_list:
            pts = np.array(eval(cell['contour']),dtype=int)
            x0,x1 = pts[:,0].min(), pts[:,0].max()
            y0,y1 = pts[:,1].min(), pts[:,1].max()
            g0[y0:y1+1, x0:x1+1] = True
            g1[y0:y1+1, x0:x1+1] = True

        def astar(sx,sy,sl, tx,ty):
            """
            简易 A* on 2-layer grid，每步水平/垂直 cost=1，via cost=1。
            """
            hq = [(0,sx,sy,sl)]
            dist = {(sx,sy,sl):0}
            while hq:
                f,x,y,l = heapq.heappop(hq)
                g = dist[(x,y,l)]
                if (x,y,l)==(tx,ty,sl):
                    return g
                for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx,ny,nl = x+dx, y+dy, l
                    if not (0<=nx<self.grid_size and 0<=ny<self.grid_size): continue
                    if (g0 if nl==0 else g1)[ny,nx]: continue
                    ng = g+1
                    if dist.get((nx,ny,nl),1e9)>ng:
                        dist[(nx,ny,nl)] = ng
                        h = abs(nx-tx)+abs(ny-ty)
                        heapq.heappush(hq,(ng+h,nx,ny,nl))
                # via
                ol = 1-l
                if not (g0 if ol==0 else g1)[y,x]:
                    ng = g+1
                    if dist.get((x,y,ol),1e9)>ng:
                        dist[(x,y,ol)] = ng
                        h = abs(x-tx)+abs(y-ty)
                        heapq.heappush(h,(ng+h,x,y,ol))
            return 1e4  # 无解惩罚

        # 累加每个 net 的 A* 成对路由
        total = 0.0
        for net in self.net_list:
            plist = net['pinList']
            for i in range(len(plist)-1):
                c1,p1 = plist[i]['cellName'], plist[i]['pinName']
                c2,p2 = plist[i+1]['cellName'], plist[i+1]['pinName']
                # 找坐标
                coord = {}
                for cell in self.cell_list:
                    if cell['cellName'] in (c1,c2):
                        for pin in cell['pinList']:
                            coord[(cell['cellName'],pin['pinName'])] = \
                              np.array(eval(pin['center']),dtype=int)
                (x1,y1), (x2,y2) = coord[(c1,p1)], coord[(c2,p2)]
                total += astar(x1,y1,0, x2,y2)
        return -total
