import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_pcb_image(fused_state, pcb_key):
    image = fused_state[pcb_key]['image']
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title(f'PCB Image: {pcb_key}')
    plt.axis('on')
    plt.show()

def visualize_pcb_graph(fused_state, pcb_key):
    graph = fused_state[pcb_key]['graph']
    node_feats = graph["node_feats"]
    edges = graph["edges"]
    nodes = graph["nodes"]

    G = nx.Graph()
    for i, node in enumerate(nodes):
        x, y = node_feats[i][1], node_feats[i][2]
        G.add_node(i, label=f"{node[0]}:{node[1]}", pos=(x, y))
    for edge in edges:
        G.add_edge(edge[0], edge[1])

    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=100, node_color='orange', with_labels=False, edge_color='gray', alpha=0.7)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title(f'PCB GCN Graph: {pcb_key}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('on')
    plt.show()

def print_sequence_tokens(fused_state, pcb_key, max_lines=20):
    seq = fused_state[pcb_key]['sequence']
    print(f"\n[Transformer sequence] for {pcb_key} (show up to {max_lines} lines):")
    for i, line in enumerate(seq):
        print(' '.join(str(token) for token in line))
        if i+1 >= max_lines:
            print("... (truncated)")
            break

if __name__ == "__main__":
    with open('pcb_fused_rl_state.pkl', 'rb') as f:
        fused_state = pickle.load(f)
    pcb_keys = list(fused_state.keys())
    print(f"Available PCB layouts: {pcb_keys}")
    pcb_key = input("请输入要可视化的PCB json文件名（留空则选第一个）: ").strip()
    if pcb_key == '' or pcb_key not in fused_state:
        pcb_key = pcb_keys[0]

    print("选择功能:")
    print("1 - 可视化 image（灰度布局图）")
    print("2 - 可视化 graph（GCN节点/连线）")
    print("3 - 打印 sequence（Transformer token序列）")
    func = input("输入序号（1/2/3, 可多选如'12'）: ").strip()
    if "1" in func:
        visualize_pcb_image(fused_state, pcb_key)
    if "2" in func:
        visualize_pcb_graph(fused_state, pcb_key)
    if "3" in func:
        print_sequence_tokens(fused_state, pcb_key)
