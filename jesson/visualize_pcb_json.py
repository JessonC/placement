import os
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import pickle



def load_pcb_jsons(folder="pcb_cell_jsons"):
    pcb_dict = {}
    for fname in os.listdir(folder):
        if fname.endswith(".json"):
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                pcb_dict[fname] = json.load(f)
    print(f"Loaded {len(pcb_dict)} PCB jsons from '{folder}'")
    return pcb_dict

def color_palette(n):
    # 随机调色盘，不重复
    import matplotlib
    colors = list(matplotlib.colormaps['tab20'].colors)
    random.shuffle(colors)
    while len(colors) < n:
        colors += colors
    return colors[:n]

def plot_pcb(pcb_data, title="PCB"):
    cell_list = pcb_data["cellList"]
    net_list = pcb_data.get("netList", [])
    fig, ax = plt.subplots(figsize=(8,8))
    color_map = {}
    colors = color_palette(len(cell_list))
    # 画器件
    for idx, cell in enumerate(cell_list):
        c = colors[idx]
        contour = eval(cell["contour"])
        xs = [pt[0] for pt in contour] + [contour[0][0]]
        ys = [pt[1] for pt in contour] + [contour[0][1]]
        ax.plot(xs, ys, color=c, linewidth=2)
        cell_center = eval(cell["center"])
        ax.text(cell_center[0], cell_center[1], cell["cellName"],
                color=c, fontsize=9, ha='center', va='center', bbox=dict(facecolor='white', edgecolor=c, boxstyle='round,pad=0.3', alpha=0.7))
        # 画pin点
        for pin in cell["pinList"]:
            pcenter = eval(pin["center"])
            ax.plot(pcenter[0], pcenter[1], marker='o', color=c, markersize=4, alpha=0.7)
        color_map[cell["cellName"]] = c
    # 画net连线
    for net in net_list:
        pinList = net["pinList"]
        if len(pinList) < 2:
            continue
        pin_coords = []
        for pin in pinList:
            # 找pin坐标
            cell = next((c for c in cell_list if c["cellName"]==pin["cellName"]), None)
            if cell:
                pinitem = next((p for p in cell["pinList"] if p["pinName"]==pin["pinName"]), None)
                if pinitem:
                    pin_coords.append(eval(pinitem["center"]))
        # 画连线（全部连成一串）
        for i in range(len(pin_coords)-1):
            x1, y1 = pin_coords[i]
            x2, y2 = pin_coords[i+1]
            ax.plot([x1, x2], [y1, y2], color='gray', linestyle='-', linewidth=1, alpha=0.8)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder = "pcb_pre_jsons"
    pcb_dict = load_pcb_jsons(folder)
    for fname, pcb_data in pcb_dict.items():
        plot_pcb(pcb_data, title=fname)
    # with open("all_pcb_dict.pkl", "wb") as f:
    #     pickle.dump(pcb_dict, f)
    # print("Saved all PCB dict to all_pcb_dict.pkl")
