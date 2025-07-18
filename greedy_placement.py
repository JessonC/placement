import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json
import random
from copy import deepcopy

def load_all_pcb(filename="all_pcb_dict.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

def find_pin(cell, pinName):
    for pin in cell["pinList"]:
        if pin["pinName"] == pinName:
            return pin
    raise ValueError(f"Pin {pinName} not found in {cell['cellName']}")

def get_chip_area(cell):
    contour = eval(cell["contour"])
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]
    return (max(xs)-min(xs)) * (max(ys)-min(ys))

def aabb_overlap(box1, box2):
    return not (box1[2] <= box2[0] or box1[0] >= box2[2] or box1[3] <= box2[1] or box1[1] >= box2[3])

def cell_bbox(center, w, h):
    cx, cy = center
    return [cx-w//2, cy-h//2, cx+w//2, cy+h//2]

def plot_layout(cells, placed_positions, show=True):
    fig, ax = plt.subplots(figsize=(8,8))
    for cell, pos in zip(cells, placed_positions):
        center = np.array(eval(cell["center"])) + np.array(pos)
        w = abs(eval(cell["contour"])[0][0] - eval(cell["contour"])[1][0])
        h = abs(eval(cell["contour"])[0][1] - eval(cell["contour"])[3][1])
        rect = patches.Rectangle((center[0]-w/2, center[1]-h/2), w, h, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        ax.text(center[0], center[1], cell["cellName"], color='r', ha='center', va='center')
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_aspect('equal')
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()

def greedy_layout(pcb_data):
    cellList = deepcopy(pcb_data["cellList"])
    netList = pcb_data["netList"]
    cell_areas = [get_chip_area(cell) for cell in cellList]
    cell_names = [cell["cellName"] for cell in cellList]

    # 统计主芯片
    link_count = {name: 0 for name in cell_names}
    for net in netList:
        linked_cells = set([pin["cellName"] for pin in net["pinList"]])
        for name in linked_cells:
            link_count[name] += len(linked_cells)-1
    score = [a*0.6 + link_count[n]*0.4 for a, n in zip(cell_areas, cell_names)]
    main_idx = np.argmax(score)
    main_chip = cellList[main_idx]
    other_cells = [cell for i,cell in enumerate(cellList) if i != main_idx]
    placed_cells = [main_chip]
    placed_positions = [np.array([127,127])-np.array(eval(main_chip["center"]))]
    occupied_boxes = [
        cell_bbox(np.array([127,127]),
                  abs(eval(main_chip["contour"])[0][0] - eval(main_chip["contour"])[1][0]),
                  abs(eval(main_chip["contour"])[0][1] - eval(main_chip["contour"])[3][1]))
    ]
    placed_pins_dict = {}  # { (cellName, pinName): 坐标(已加偏移) }
    # 初始化主芯片pin实际坐标
    for pin in main_chip["pinList"]:
        placed_pins_dict[(main_chip["cellName"], pin["pinName"])] = (np.array(eval(pin["center"])) + placed_positions[0])

    def get_linked_pin_pairs(cell):
        # 找与已放置器件有net连接的pin-pair (待放cell的pin, 已放cell的pin)
        linked_pairs = []
        cellname = cell["cellName"]
        for net in netList:
            pinList = net["pinList"]
            # 只考虑待放cell和已放的cell之间的连线
            cell_pins = [p for p in pinList if p["cellName"]==cellname]
            for pin in cell_pins:
                for p in pinList:
                    if p["cellName"] != cellname and (p["cellName"], p["pinName"]) in placed_pins_dict:
                        linked_pairs.append( (pin, p) )
        return linked_pairs

    # 按照与已放器件连线数量最多优先放
    other_cells.sort(key=lambda cell: len(get_linked_pin_pairs(cell)), reverse=True)

    for cell in other_cells:
        w = abs(eval(cell["contour"])[0][0] - eval(cell["contour"])[1][0])
        h = abs(eval(cell["contour"])[0][1] - eval(cell["contour"])[3][1])
        # 找所有与已放置器件有net连接的pin对
        linked_pairs = get_linked_pin_pairs(cell)
        if not linked_pairs:
            # 没有与已放器件的连线，随机找可放区域
            for _ in range(100):
                cx = random.randint(w//2, 255-w//2)
                cy = random.randint(h//2, 255-h//2)
                new_box = [cx-w//2, cy-h//2, cx+w//2, cy+h//2]
                if not any(aabb_overlap(new_box, obox) for obox in occupied_boxes):
                    best_offset = np.array([cx,cy])-np.array(eval(cell["center"]))
                    break
            else:
                best_offset = np.zeros(2)
        else:
            # 在若干候选点中找最短连线方案
            best_score = float("inf")
            best_offset = None
            # 候选点：与目标pin连线时，器件应如何偏移
            all_target_coords = []
            for pin, peer in linked_pairs:
                peer_coord = placed_pins_dict[(peer["cellName"], peer["pinName"])]
                all_target_coords.append(peer_coord)
            # 网格采样候选点，或以target pin为圆心做采样
            for ref in all_target_coords:
                for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                    for r in [10, 20, 30]:
                        cx = int(ref[0] + np.cos(angle)*r)
                        cy = int(ref[1] + np.sin(angle)*r)
                        if cx-w//2<0 or cy-h//2<0 or cx+w//2>255 or cy+h//2>255:
                            continue
                        offset = np.array([cx,cy])-np.array(eval(cell["center"]))
                        # 计算该offset下所有相关pin的欧氏距离之和
                        total_dist = 0
                        for pin, peer in linked_pairs:
                            pin_local = np.array(eval(find_pin(cell, pin["pinName"])["center"])) + offset
                            # 对方已放器件pin坐标直接查dict
                            peer_pin = placed_pins_dict[(peer["cellName"], peer["pinName"])]
                            total_dist += np.linalg.norm(pin_local - peer_pin)
                        # 保证不重叠
                        new_box = [cx-w//2, cy-h//2, cx+w//2, cy+h//2]
                        if not any(aabb_overlap(new_box, obox) for obox in occupied_boxes):
                            if total_dist < best_score:
                                best_score = total_dist
                                best_offset = offset
            if best_offset is None:
                # fallback随机无碰撞点
                for _ in range(200):
                    cx = random.randint(w//2, 255-w//2)
                    cy = random.randint(h//2, 255-h//2)
                    new_box = [cx-w//2, cy-h//2, cx+w//2, cy+h//2]
                    if not any(aabb_overlap(new_box, obox) for obox in occupied_boxes):
                        best_offset = np.array([cx,cy])-np.array(eval(cell["center"]))
                        break
                if best_offset is None:
                    best_offset = np.zeros(2)

        placed_cells.append(cell)
        placed_positions.append(best_offset)
        # 更新所有pin的全局坐标
        for pin in cell["pinList"]:
            placed_pins_dict[(cell["cellName"], pin["pinName"])] = np.array(eval(pin["center"])) + best_offset
        cx, cy = np.array(eval(cell["center"])) + best_offset
        occupied_boxes.append([cx-w//2, cy-h//2, cx+w//2, cy+h//2])
    # plot_layout(placed_cells, placed_positions)
    return placed_cells, placed_positions


def update_cell_positions(cellList, placed_positions):
    new_cell_list = []
    for cell, pos in zip(cellList, placed_positions):
        old_center = np.array(eval(cell["center"]))
        new_center = list((old_center + pos).astype(int))
        delta = np.array(new_center) - old_center

        cell = deepcopy(cell)
        cell["center"] = str(new_center)

        # 更新contour
        old_contour = eval(cell["contour"])
        new_contour = [list((np.array(pt) + delta).astype(int)) for pt in old_contour]
        cell["contour"] = str(new_contour)

        # 更新pin
        new_pinList = []
        for pin in cell["pinList"]:
            old_pin_center = np.array(eval(pin["center"]))
            new_pin_center = list((old_pin_center + delta).astype(int))
            old_pin_contour = eval(pin["contour"])
            new_pin_contour = [list((np.array(pt) + delta).astype(int)) for pt in old_pin_contour]
            new_pin = deepcopy(pin)
            new_pin["center"] = str(new_pin_center)
            new_pin["contour"] = str(new_pin_contour)
            new_pinList.append(new_pin)
        cell["pinList"] = new_pinList
        new_cell_list.append(cell)
    return new_cell_list


def save_pre_layout_json(filename, cellList, netList, outdir="pcb_pre_jsons"):
    os.makedirs(outdir, exist_ok=True)
    save_path = os.path.join(outdir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"cellList": cellList, "netList": netList}, f, indent=2, ensure_ascii=False)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    all_pcb = load_all_pcb("all_pcb_dict.pkl")
    os.makedirs("pcb_pre_jsons", exist_ok=True)
    for pcb_name, pcb_data in all_pcb.items():
        placed_cells, placed_positions = greedy_layout(pcb_data)
        # 更新center位置
        new_cell_list = update_cell_positions(placed_cells, placed_positions)
        # 保存到新文件夹
        save_pre_layout_json(pcb_name, new_cell_list, pcb_data["netList"], outdir="pcb_pre_jsons")
        # 可选：可视化，不想批量弹窗就注释下一行
        # plot_layout(placed_cells, placed_positions, show=True)
