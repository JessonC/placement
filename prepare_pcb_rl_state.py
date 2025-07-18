import os
import json
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from placement_game import PCBRLEnv

# 辅助函数安全获取pin的center
def get_pin_center(cell_list, cell_name, pin_name):
    cell = next((c for c in cell_list if c["cellName"] == cell_name), None)
    if cell is None:
        raise RuntimeError(f"cell '{cell_name}' not found!")
    pin = next((p for p in cell["pinList"] if p["pinName"] == pin_name), None)
    if pin is None:
        raise RuntimeError(f"pin '{pin_name}' not found in cell '{cell_name}'!")
    if "center" not in pin:
        raise RuntimeError(f"pin '{pin_name}' in cell '{cell_name}' missing 'center'!")
    return np.array(eval(pin["center"]))

# 图信息（GCN）
def build_graph_info(env):
    pin2id = {}
    nodes = []
    edges = []
    node_feats = []
    cell_name2id = {cell["cellName"]: idx for idx, cell in enumerate(env.cell_list)}
    idx = 0

    for net in env.net_list:
        pinList = net["pinList"]
        if len(pinList) < 2:
            continue
        for pin in pinList:
            key = (pin["cellName"], pin["pinName"])
            if key not in pin2id:
                pin2id[key] = idx
                cell_idx = cell_name2id[pin["cellName"]]
                pin_center = get_pin_center(env.cell_list, pin["cellName"], pin["pinName"])
                coord = pin_center / 255.0  # 归一化
                node_feats.append([cell_idx, coord[0], coord[1]])
                nodes.append(key)
                idx += 1
        for i in range(len(pinList)):
            for j in range(i+1, len(pinList)):
                edges.append((pin2id[(pinList[i]["cellName"], pinList[i]["pinName"])],
                              pin2id[(pinList[j]["cellName"], pinList[j]["pinName"])]))
                edges.append((pin2id[(pinList[j]["cellName"], pinList[j]["pinName"])],
                              pin2id[(pinList[i]["cellName"], pinList[i]["pinName"])]))
    return {
        "nodes": nodes,
        "node_feats": np.array(node_feats, dtype=np.float32),
        "edges": np.array(edges, dtype=np.int64),
    }

# 图像信息（CNN/ResNet）
def build_image_info(env):
    img = np.zeros((256, 256), dtype=np.float32)
    num_cells = len(env.cell_list)
    for idx, cell in enumerate(env.cell_list):
        contour = np.array(eval(cell["contour"]), dtype=np.int32)
        gray_val = (idx+1) / num_cells
        cv2.fillPoly(img, [contour], color=gray_val)
    img = np.clip(img, 0, 1)
    return img

# 序列信息（Transformer）
def build_vocab(cell_list, max_distance=256):
    vocab = {"SOS": 0, "EOS": 1, "CROSS": 2}
    for idx, cell in enumerate(cell_list):
        vocab[cell["cellName"]] = 10 + idx
    for d in range(max_distance+1):
        vocab[f"DIST_{d}"] = 1000 + d
    return vocab

def build_sequence_info(env, vocab):
    sequence = []
    for net in env.net_list:
        pinList = net["pinList"]
        if len(pinList) < 2:
            continue
        for i, pin in enumerate(pinList):
            cell_name = pin["cellName"]
            pin_name = pin["pinName"]
            pin_center = get_pin_center(env.cell_list, cell_name, pin_name)
            for j, peer in enumerate(pinList):
                if i == j:
                    continue
                peer_center = get_pin_center(env.cell_list, peer["cellName"], peer["pinName"])
                dist = int(np.linalg.norm(pin_center - peer_center))
                seq = [
                    vocab["SOS"],
                    vocab[cell_name],
                    vocab.get(pin_name, 2000),
                    vocab.get(f"DIST_{dist}", 1000+min(dist, 256)),
                    vocab["EOS"]
                ]
                sequence.append(seq)
    return np.array(sequence, dtype=np.int64)

def build_sequence_info_v2(env):
    SOS, SOE, MOS, MOE = 0, 1, 2, 3
    token_seq = [SOS]

    # 获取graph信息，用于node编号对应
    graph_info = build_graph_info(env)
    node_key2idx = {key: idx for idx, key in enumerate(graph_info["nodes"])}

    # 构建器件名字到ID映射 (从10开始)
    cell_name2idx = {cell["cellName"]: idx + 10 for idx, cell in enumerate(env.cell_list)}

    for cell in env.cell_list:
        module_tokens = []
        cell_idx = cell_name2idx[cell["cellName"]]

        # 当前模块中的所有pin对应的graph node id
        module_nodes = []
        for pin in cell["pinList"]:
            key = (cell["cellName"], pin["pinName"])
            if key in node_key2idx:
                module_nodes.append(node_key2idx[key])

        # 模块内有效的连接关系
        for net in env.net_list:
            pins_in_net = net["pinList"]
            # 找到当前net中属于该模块的pin
            pins_in_cell = [p for p in pins_in_net if p["cellName"] == cell["cellName"]]
            pins_outside_cell = [p for p in pins_in_net if p["cellName"] != cell["cellName"]]

            # 只处理模块内的pin与模块外的pin之间的连线
            for p_in in pins_in_cell:
                key_in = (p_in["cellName"], p_in["pinName"])
                if key_in not in node_key2idx:
                    continue
                node_in = node_key2idx[key_in]
                for p_out in pins_outside_cell:
                    key_out = (p_out["cellName"], p_out["pinName"])
                    if key_out not in node_key2idx:
                        continue
                    node_out = node_key2idx[key_out]

                    # 计算两点距离
                    pin_in_center = get_pin_center(env.cell_list, *key_in)
                    pin_out_center = get_pin_center(env.cell_list, *key_out)
                    dist = int(np.linalg.norm(pin_in_center - pin_out_center))

                    module_tokens.extend([node_in + 100, node_out + 100, dist])

        # 如果当前模块有连线关系，则加入序列
        if module_tokens:
            token_seq.extend([MOS, cell_idx])
            token_seq.extend(module_tokens)
            token_seq.append(MOE)

    token_seq.append(SOE)
    return np.array(token_seq, dtype=np.int32)


# 融合state
def get_fused_state(env):
    graph_info = build_graph_info(env)
    image_info = build_image_info(env)
    vocab = build_vocab(env.cell_list)
    sequence_info = build_sequence_info_v2(env)
    return {
        "graph": graph_info,
        "image": image_info,
        "sequence": sequence_info
    }

# 主处理流程
def main():
    data_dir = "pcb_pre_jsons"
    save_pkl = "pcb_fused_rl_state.pkl"
    all_state = {}
    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    for fname in tqdm(files, desc="Processing PCB"):
        pcb_path = os.path.join(data_dir, fname)
        try:
            env = PCBRLEnv(pcb_path)
            fused_state = get_fused_state(env)
            all_state[fname] = fused_state
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            import traceback
            traceback.print_exc()
    with open(save_pkl, "wb") as f:
        pickle.dump(all_state, f)
    print(f"\nAll states saved to {save_pkl}.")

if __name__ == "__main__":
    main()
