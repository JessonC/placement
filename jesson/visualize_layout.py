import json, numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def safe_load(x):
    return json.loads(x) if isinstance(x, str) else x

# 1. 读入 best_layout.json
data = json.load(open('best_layout.json', 'r'))
cells, nets = data['cellList'], data['netList']

# 2. 新建画布
fig, ax = plt.subplots(figsize=(6, 6))

# 3. 画器件轮廓和名称
for c in cells:
    pts = np.array(safe_load(c['contour']), int)
    ax.plot(*np.vstack([pts, pts[0]]).T, '-', linewidth=2)
    x, y = safe_load(c['center'])
    ax.text(x, y, c['cellName'], ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

# 4. 画飞线
for net in nets:
    coords = []
    for p in net['pinList']:
        for c in cells:
            if c['cellName'] == p['cellName']:
                for pp in c['pinList']:
                    if pp['pinName'] == p['pinName']:
                        coords.append(np.array(safe_load(pp['center']), int))
                        break
                break
    if len(coords) > 1:
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        ax.plot(xs, ys, '--', color='gray', alpha=0.7)

ax.set_xlim(0, 256)
ax.set_ylim(0, 256)
ax.set_aspect('equal')
plt.title("PCB_Layout")
plt.show()
