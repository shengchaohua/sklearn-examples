import numpy as np
import matplotlib.pyplot as plt

l1_color = 'navy'
l2_color = 'c'
elastic_net_color = 'darkorange'

line = np.linspace(-1.5, 1.5, 1001)
xx, yy = np.meshgrid(line, line)

l1 = np.abs(xx) + np.abs(yy)
l2 = xx ** 2 + yy ** 2
l1_ratio = 0.5
elastic_net = l1_ratio * l1 + (1 - l1_ratio) * l2

plt.figure(figsize=(10, 10), dpi=100)
ax = plt.gca()

l1_contour = plt.contour(xx, yy, l1, levels=[1], colors=l1_color)
l2_contour = plt.contour(xx, yy, l2, levels=[1], colors=l2_color)
elastic_net_contour = plt.contour(xx, yy, elastic_net, levels=[1],
                                  colors=elastic_net_color)
ax.set_aspect('equal')
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')

plt.clabel(l1_contour, inline=1, fontsize=18,
           fmt={1.0: 'L1'}, manual=[(-1, -1)])
plt.clabel(l2_contour, inline=1, fontsize=18,
           fmt={1.0: 'L2'}, manual=[(-1, -1)])
plt.clabel(elastic_net_contour, inline=1, fontsize=18,
           fmt={1.0: 'elastic-net'}, manual=[(-1, -1)])

plt.tight_layout()
plt.show()
