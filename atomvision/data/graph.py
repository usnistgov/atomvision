"""Module for Graph Torch Dataset Class."""
import torch
import dgl
from typing import Tuple, List

# import typer
import numpy as np


def prepare_line_graph_batch(
    batch: Tuple[Tuple[dgl.DGLGraph, dgl.DGLGraph], torch.Tensor],
    device=None,
    non_blocking=False,
):
    """Send line graph batch to device."""
    # Note: the batch is a nested tuple, with the graph and line graph together
    g, lg, t = batch
    batch = (
        (
            g.to(device, non_blocking=non_blocking),
            lg.to(device, non_blocking=non_blocking),
        ),
        t.to(device, non_blocking=non_blocking),
    )
    # print("tensor device", device)
    return batch


class GraphDataset(torch.utils.data.Dataset):
    """Module for Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        ids=[],
        graphs=[],
        line_graphs=[],
        labels=[],  # 1,2,3,4,5 etc
        transform=None,
        line_graph=True,
        classification=True,
        id_tag="jid",
    ):
        """Get Pytorch Dataset for atomistic graphs."""
        # `df`: pandas dataframe from e.g. jarvis.db.figshare.data
        # `graphs`: DGLGraph representations corresponding to rows in `df`
        # `target`: key for label column in `df`
        self.ids = ids
        self.graphs = graphs
        self.line_graphs = line_graphs
        if not self.ids:
            self.ids = [str(j) for j in np.arange(len(self.graphs))]
        # self.targets = target
        self.line_graph = line_graph
        self.labels = labels
        # (OneHotEncoder(sparse=False).fit_transform(np.array(labels).reshape(-1,1)))
        # self.labels = labels
        self.labels = torch.tensor(self.labels).type(torch.get_default_dtype())
        self.transform = transform

        self.prepare_batch = prepare_line_graph_batch

        if classification:
            self.labels = self.labels.view(-1).long()
            print("Classification dataset.", self.labels)

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        # print ('GRAPHS',len(self.graphs),idx)
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        if self.line_graph:
            return g, self.line_graphs[idx], label

        return g, label

    @staticmethod
    def collate(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)

    @staticmethod
    def collate_line_graph(
        samples: List[Tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor]]
    ):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, torch.tensor(labels)
