import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def visualize_pcb_image(fused_state, pcb_key=None):
    # fused_state: dict, key为json文件名，值为三模态dict
    if pcb_key is None:
        pcb_key = list(fused_state.keys())[0]
    image = fused_state[pcb_key]['image']  # shape: [256, 256]
    plt.figure(figsize=(6,6))
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title(f'PCB Image: {pcb_key}')
    plt.axis('on')
    plt.show()

if __name__ == "__main__":
    with open('pcb_fused_rl_state.pkl', 'rb') as f:
        fused_state = pickle.load(f)
    print(f"Available PCB layouts: {list(fused_state.keys())}")
    pcb_key = input("请输入要可视化的PCB json文件名（留空则选第一个）: ").strip()
    if pcb_key == '' or pcb_key not in fused_state:
        pcb_key = list(fused_state.keys())[0]
    visualize_pcb_image(fused_state, pcb_key)
