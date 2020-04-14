import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ---- NetworkX compatibility
def node_iter(G):
    if float(nx.__version__)<2.0:
        return G.nodes()
    else:
        return G.nodes

def node_dict(G):
    if float(nx.__version__)>2.1:
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict
# ---------------------------


def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, origin=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=arr.shape[::-1], dpi=1, frameon=False)
    canvas = FigureCanvas(fig)
    fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=1, format=format)

def plot_graph(plt, G):
    plt.title('num of nodes: '+str(G.number_of_nodes()), fontsize = 4)
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
    plt.axis("off")
    pos = nx.spring_layout(G)
    # pos = nx.spectral_layout(G)
    nx.draw_networkx(G, with_labels=True, node_size=4, width=0.3, font_size = 3, node_color=colors,pos=pos)

def draw_graph_list(G_list, row, col, fname = 'figs/test'):
    # draw graph view
    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row,col,i+1)
        plot_graph(plt, G)
        
    plt.tight_layout()
    plt.savefig(fname+'_view.png', dpi=600)
    plt.close()

    # draw degree distribution
    plt.switch_backend('agg')
    for i, G in enumerate(G_list):
        plt.subplot(row, col, i + 1)
        G_deg = np.array(list(G.degree(G.nodes()).values()))
        bins = np.arange(20)
        plt.hist(np.array(G_deg), bins=bins, align='left')
        plt.xlabel('degree', fontsize = 3)
        plt.ylabel('count', fontsize = 3)
        G_deg_mean = 2*G.number_of_edges()/float(G.number_of_nodes())
        plt.title('average degree: {:.2f}'.format(G_deg_mean), fontsize=4)
        plt.tick_params(axis='both', which='major', labelsize=3)
        plt.tick_params(axis='both', which='minor', labelsize=3)
    plt.tight_layout()
    plt.savefig(fname+'_degree.png', dpi=600)
    plt.close()

    # degree_sequence = sorted(nx.degree(G).values(), reverse=True)  # degree sequence
    # plt.loglog(degree_sequence, 'b-', marker='o')
    # plt.title("Degree rank plot")
    # plt.ylabel("degree")
    # plt.xlabel("rank")
    # plt.savefig('figures/degree_view_' + prefix + '.png', dpi=200)
    # plt.close()

    # draw clustering distribution
    #plt.switch_backend('agg')
    #for i, G in enumerate(G_list):
    #    plt.subplot(row, col, i + 1)
    #    G_cluster = list(nx.clustering(G).values())
    #    bins = np.linspace(0,1,20)
    #    plt.hist(np.array(G_cluster), bins=bins, align='left')
    #    plt.xlabel('clustering coefficient', fontsize=3)
    #    plt.ylabel('count', fontsize=3)
    #    G_cluster_mean = sum(G_cluster) / len(G_cluster)
    #    # if i % 2 == 0:
    #    #     plt.title('real average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #    # else:
    #    #     plt.title('pred average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #    plt.title('average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #    plt.tick_params(axis='both', which='major', labelsize=3)
    #    plt.tick_params(axis='both', which='minor', labelsize=3)
    #plt.tight_layout()
    #plt.savefig(fname+'_clustering.png', dpi=600)
    #plt.close()

    ## draw circle distribution
    #plt.switch_backend('agg')
    #for i, G in enumerate(G_list):
    #    plt.subplot(row, col, i + 1)
    #    cycle_len = []
    #    cycle_all = nx.cycle_basis(G)
    #    for item in cycle_all:
    #        cycle_len.append(len(item))

    #    bins = np.arange(20)
    #    plt.hist(np.array(cycle_len), bins=bins, align='left')
    #    plt.xlabel('cycle length', fontsize=3)
    #    plt.ylabel('count', fontsize=3)
    #    G_cycle_mean = 0
    #    if len(cycle_len)>0:
    #        G_cycle_mean = sum(cycle_len) / len(cycle_len)
    #    # if i % 2 == 0:
    #    #     plt.title('real average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #    # else:
    #    #     plt.title('pred average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #    plt.title('average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #    plt.tick_params(axis='both', which='major', labelsize=3)
    #    plt.tick_params(axis='both', which='minor', labelsize=3)
    #plt.tight_layout()
    #plt.savefig(fname+'_cycle.png', dpi=600)
    #plt.close()

    ## draw community distribution
    #plt.switch_backend('agg')
    #for i, G in enumerate(G_list):
    #    plt.subplot(row, col, i + 1)
    #    parts = community.best_partition(G)
    #    values = np.array([parts.get(node) for node in G.nodes()])
    #    counts = np.sort(np.bincount(values)[::-1])
    #    pos = np.arange(len(counts))
    #    plt.bar(pos,counts,align = 'edge')
    #    plt.xlabel('community ID', fontsize=3)
    #    plt.ylabel('count', fontsize=3)
    #    G_community_count = len(counts)
    #    # if i % 2 == 0:
    #    #     plt.title('real average clustering: {}'.format(G_community_count), fontsize=4)
    #    # else:
    #    #     plt.title('pred average clustering: {}'.format(G_community_count), fontsize=4)
    #    plt.title('average clustering: {}'.format(G_community_count), fontsize=4)
    #    plt.tick_params(axis='both', which='major', labelsize=3)
    #    plt.tick_params(axis='both', which='minor', labelsize=3)
    #plt.tight_layout()
    #plt.savefig(fname+'_community.png', dpi=600)
    #plt.close()


def exp_moving_avg(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a


