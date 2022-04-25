"""Module for plotting edges."""
import networkx as nx
import matplotlib.pyplot as plt


def plot_edges(g: nx.Graph, ax=None):
    """Plot edges."""
    if ax is None:
        ax = plt.gca()

    for (i, j) in g.edges:
        u = g.nodes[i]
        v = g.nodes[j]

        # convert coordinates back to pixel scale
        # angstrom / (angstrom/px) -> px
        ru = u["pos"] / g.graph["px_angstrom"]
        rv = v["pos"] / g.graph["px_angstrom"]

        ax.plot([ru[0], rv[0]], [ru[1], rv[1]], "-w")
        # ax.plot([u["pos"][0], v["pos"][0]], [u["pos"][1], v["pos"][1]], "-w")
