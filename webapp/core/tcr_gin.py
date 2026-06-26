#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TCR-GIN model definition used by the standalone Streamlit app."""

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_mean_pool


class TCR_GIN(nn.Module):
    """TCR-GIN architecture matching the checkpoints used by the webapp."""

    def __init__(self, args):
        super(TCR_GIN, self).__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.jk_type = args.jk_type
        self.use_virtual_node = args.use_virtual_node
        self.use_residual = args.use_residual

        act_name = args.activation_fn.lower()
        if act_name == "gelu":
            self.activation_module = nn.GELU()
        elif act_name == "relu":
            self.activation_module = nn.ReLU()
        elif act_name in ("sigmoid", "sigmod"):
            self.activation_module = nn.Sigmoid()
        else:
            raise ValueError(f"{args.activation_fn} error: only 'gelu', 'relu', 'sigmoid' are supported")

        self.input_proj = nn.Linear(args.input_dim, self.hidden_dim)

        if self.use_virtual_node:
            self.virtual_node_embedding = nn.Embedding(1, self.hidden_dim)
            self.mlp_virtual_node_list = nn.ModuleList()
            for _ in range(self.num_layers):
                self.mlp_virtual_node_list.append(
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                        nn.BatchNorm1d(self.hidden_dim * 2),
                        self.activation_module,
                        nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                        nn.BatchNorm1d(self.hidden_dim),
                        self.activation_module,
                    )
                )

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(self.make_gin_conv(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        self.pool = global_mean_pool

        if self.jk_type in ["cat", "max", "lstm"]:
            self.jk_layer = JumpingKnowledge(
                mode=self.jk_type,
                channels=self.hidden_dim,
                num_layers=self.num_layers,
            )
        elif self.jk_type == "last":
            self.jk_layer = None
        else:
            raise ValueError(f"jk_type error: '{self.jk_type}'")

        if self.jk_type == "cat":
            mlp_input_dim = self.hidden_dim * self.num_layers
        else:
            mlp_input_dim = self.hidden_dim

        self.prediction_head = nn.Sequential(
            nn.Linear(mlp_input_dim, self.hidden_dim),
            self.activation_module,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def make_gin_conv(self, input_dim, output_dim):
        return GINConv(
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                self.activation_module,
                nn.Linear(output_dim, output_dim),
            )
        )

    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        h = self.input_proj(x)

        if self.use_virtual_node:
            virtual_node_feat = self.virtual_node_embedding(
                torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device)
            )

        layer_outputs = []
        for i in range(self.num_layers):
            if self.use_virtual_node:
                h = h + virtual_node_feat[batch]

            h_prev = h
            h = self.convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = self.activation_module(h)

            if self.use_residual and h.shape == h_prev.shape:
                h = h + h_prev

            layer_outputs.append(h)

            if self.use_virtual_node and i < self.num_layers - 1:
                aggregated_graph_feat = global_add_pool(h, batch)
                virtual_node_feat = virtual_node_feat + self.mlp_virtual_node_list[i](aggregated_graph_feat)

        if self.jk_layer is not None:
            node_representation = self.jk_layer(layer_outputs)
        else:
            node_representation = layer_outputs[-1]

        graph_representation = self.pool(node_representation, batch)
        prediction = self.prediction_head(graph_representation)

        return prediction.view(-1, 1)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
