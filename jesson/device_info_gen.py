import json
import random
import os
import numpy as np

def random_rect_contour(center, w, h):
    cx, cy = center
    return [
        [cx - w // 2, cy - h // 2],
        [cx + w // 2, cy - h // 2],
        [cx + w // 2, cy + h // 2],
        [cx - w // 2, cy + h // 2]
    ]


def random_pin(center):
    w, h = 2, 2
    cx, cy = center
    return [
        [cx - 1, cy - 1],
        [cx + 1, cy - 1],
        [cx + 1, cy + 1],
        [cx - 1, cy + 1]
    ]


def random_layer():
    return random.choice(['top', 'bottom'])


def random_rotation():
    return random.choice([0, 90, 180, 270])


def random_value():
    return str(random.randint(1, 100))


def gen_main_chip():
    # 主芯片
    w, h = random.randint(40, 70), random.randint(40, 70)
    cx, cy = random.randint(w//2, 255-w//2), random.randint(h//2, 255-h//2)
    center = [cx, cy]
    contour = random_rect_contour(center, w, h)
    layer = random_layer()
    rotation = random_rotation()
    value = random_value()
    npins = random.randint(80, 120)
    # 用接近正方形的网格分布
    nrow = int(np.sqrt(npins))
    ncol = int(npins / nrow)
    if nrow * ncol < npins:
        ncol += 1
    pin_positions = []
    for i in range(npins):
        row = i // ncol
        col = i % ncol
        # 均匀分布，避免贴边
        px = int(cx - w//2 + w * (col+1)/(ncol+1))
        py = int(cy - h//2 + h * (row+1)/(nrow+1))
        pin_positions.append([px, py])
    pinList = []
    for i, pin_center in enumerate(pin_positions):
        pin_contour = random_pin(pin_center)
        pin_name = f"P{i+1}"
        pinList.append({
            "center": str(pin_center),
            "contour": str(pin_contour),
            "pinName": pin_name
        })
    return {
        "cellName": "U1",
        "cellType": "MainChip",
        "center": str(center),
        "code": "mainchip",
        "contour": str(contour),
        "layer": layer,
        "pinList": pinList,
        "rotation": str(rotation),
        "value": value
    }

def gen_normal_chip(idx):
    w, h = random.randint(10, 30), random.randint(10, 30)
    cx, cy = random.randint(w//2, 255-w//2), random.randint(h//2, 255-h//2)
    center = [cx, cy]
    contour = random_rect_contour(center, w, h)
    layer = random_layer()
    rotation = random_rotation()
    value = random_value()
    npins = random.randint(2, 4)
    pin_positions = []
    if npins == 2:
        # 横向等分
        pin_positions = [
            [int(cx - w//4), cy],
            [int(cx + w//4), cy]
        ]
    elif npins == 3:
        # 水平三等分
        pin_positions = [
            [int(cx - w//4), cy],
            [cx, cy],
            [int(cx + w//4), cy]
        ]
    elif npins == 4:
        # 2x2阵列
        pin_positions = [
            [int(cx - w//4), int(cy - h//4)],
            [int(cx + w//4), int(cy - h//4)],
            [int(cx - w//4), int(cy + h//4)],
            [int(cx + w//4), int(cy + h//4)]
        ]
    else:
        # 1个pin就中心
        pin_positions = [[cx, cy]]
    pinList = []
    for i, pin_center in enumerate(pin_positions[:npins]):
        pin_contour = random_pin(pin_center)
        pin_name = f"P{i+1}"
        pinList.append({
            "center": str(pin_center),
            "contour": str(pin_contour),
            "pinName": pin_name
        })
    return {
        "cellName": f"U{idx}",
        "cellType": "NormalChip",
        "center": str(center),
        "code": "chip",
        "contour": str(contour),
        "layer": layer,
        "pinList": pinList,
        "rotation": str(rotation),
        "value": value
    }


def generate_json(filename):
    cell_list = []
    cell_list.append(gen_main_chip())
    n_other = random.randint(14, 29)
    for i in range(2, n_other + 2):
        cell_list.append(gen_normal_chip(i))

    # ==== 建立pin的全局索引结构 ====
    pin_index = {}  # (cellName, pinName) -> cell idx
    cell_pin_count = {}
    for cell in cell_list:
        cellName = cell["cellName"]
        for pin in cell["pinList"]:
            pin_index[(cellName, pin["pinName"])] = cellName
        cell_pin_count[cellName] = len(cell["pinList"])

    # ==== 主芯片直连 ====
    main_name = "U1"
    normal_names = [c["cellName"] for c in cell_list if c["cellName"] != main_name]
    num_direct = max(1, int(len(normal_names) * 0.6))
    direct_names = set(random.sample(normal_names, num_direct))
    others = [n for n in normal_names if n not in direct_names]

    netList = []
    net_id = 1
    used = set()  # 记录已连通cellName

    # 主芯片与部分芯片直连
    for cname in direct_names:
        main_p = f"P{random.randint(1, cell_pin_count[main_name])}"
        chip_p = f"P{random.randint(1, cell_pin_count[cname])}"
        netList.append({
            "netName": f"N{net_id}",
            "pinList": [
                {"cellName": main_name, "pinName": main_p},
                {"cellName": cname, "pinName": chip_p}
            ]
        })
        net_id += 1
        used.add(main_name)
        used.add(cname)

    # 其他芯片与已直连的芯片形成链式/分组互连
    for cname in others:
        # 必须和used中的一个芯片pin连
        linked_cell = random.choice(list(used))
        pin1 = f"P{random.randint(1, cell_pin_count[cname])}"
        pin2 = f"P{random.randint(1, cell_pin_count[linked_cell])}"
        netList.append({
            "netName": f"N{net_id}",
            "pinList": [
                {"cellName": cname, "pinName": pin1},
                {"cellName": linked_cell, "pinName": pin2}
            ]
        })
        net_id += 1
        used.add(cname)

    # 可选：生成若干芯片间的二级/多级连接（增加复杂性）
    if len(cell_list) > 5:
        for _ in range(random.randint(2, 6)):
            c1, c2 = random.sample(normal_names, 2)
            p1 = f"P{random.randint(1, cell_pin_count[c1])}"
            p2 = f"P{random.randint(1, cell_pin_count[c2])}"
            netList.append({
                "netName": f"N{net_id}",
                "pinList": [
                    {"cellName": c1, "pinName": p1},
                    {"cellName": c2, "pinName": p2}
                ]
            })
            net_id += 1

    # 写入json
    with open(filename, "w") as f:
        json.dump({"cellList": cell_list, "netList": netList}, f, indent=2, ensure_ascii=False)


def batch_generate(n=5, outdir="pcb_cell_jsons"):
    os.makedirs(outdir, exist_ok=True)
    for i in range(n):
        fname = os.path.join(outdir, f"pcb_cells_{i + 1}.json")
        generate_json(fname)
        print(f"Generated: {fname}")


if __name__ == "__main__":
    batch_generate(10)
