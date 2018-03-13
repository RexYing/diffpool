import networkx as nx
import numpy as np

def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, origin=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=arr.shape[::-1], dpi=1, frameon=False)
    canvas = FigureCanvas(fig)
    fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=1, format=format)

def draw_graph(G, prefix = 'test'):
    parts = community.best_partition(G)
    values = [parts.get(node) for node in G.nodes()]
    colors = []
    for i in range(len(values)):
        if values[i] == 0:
            colors.append('red')
        if values[i] == 1:
            colors.append('green')
        if values[i] == 2:
            colors.append('blue')
        if values[i] == 3:
            colors.append('yellow')
        if values[i] == 4:
            colors.append('orange')
        if values[i] == 5:
            colors.append('pink')
        if values[i] == 6:
            colors.append('black')

    # spring_pos = nx.spring_layout(G)
    plt.switch_backend('agg')
    plt.axis("off")

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, with_labels=True, node_size=35, node_color=colors,pos=pos)


    # plt.switch_backend('agg')
    # options = {
    #     'node_color': 'black',
    #     'node_size': 10,
    #     'width': 1
    # }
    # plt.figure()
    # plt.subplot()
    # nx.draw_networkx(G, **options)
    plt.savefig('figures/graph_view_'+prefix+'.png', dpi=200)
    plt.close()

    plt.switch_backend('agg')
    G_deg = nx.degree_histogram(G)
    G_deg = np.array(G_deg)
    # plt.plot(range(len(G_deg)), G_deg, 'r', linewidth = 2)
    plt.loglog(np.arange(len(G_deg))[G_deg>0], G_deg[G_deg>0], 'r', linewidth=2)
    plt.savefig('figures/degree_view_' + prefix + '.png', dpi=200)
    plt.close()

    # degree_sequence = sorted(nx.degree(G).values(), reverse=True)  # degree sequence
    # plt.loglog(degree_sequence, 'b-', marker='o')
    # plt.title("Degree rank plot")
    # plt.ylabel("degree")
    # plt.xlabel("rank")
    # plt.savefig('figures/degree_view_' + prefix + '.png', dpi=200)
    # plt.close()
