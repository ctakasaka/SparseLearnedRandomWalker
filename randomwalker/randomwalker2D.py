#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
from sksparse.cholmod import cholesky
import torch
from randomwalker import randomwalker_tools


class RandomWalker2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, elap_input, seeds_input, num_grad=1000, max_backprop=True):
        """
        num_grad: Number of sampled gradients
        max_backprop: Compute the loss only on the absolute maximum
        """
        elap = elap_input.clone().numpy()
        seeds = seeds_input.numpy()
        elap = np.squeeze(elap)

        # Building laplacian and running the RW
        pu, lap_u = randomwalker_tools.standard_RW(elap, seeds)

        # Fill seeds predictions
        p = randomwalker_tools.pu2p(pu, seeds)

        ctx.save_for_backward(seeds_input)
        ctx.lap_u = lap_u
        ctx.pu = pu
        ctx.num_grad = num_grad
        ctx.max_backprop = max_backprop

        return torch.from_numpy(p)

    @staticmethod
    def backward(ctx, grad_output):
        """
        input : grad from loss
        output: grad from the laplacian backprop
        """
        seeds_input = ctx.saved_tensors[0]
        lap_u = ctx.lap_u
        pu = ctx.pu
        num_grad = ctx.num_grad
        max_backprop = ctx.max_backprop

        gradout = grad_output.numpy()
        seeds = seeds_input.numpy()

        # Remove seeds from grad_output
        gradout = randomwalker_tools.p2pu(gradout, seeds)

        # Back propagation
        grad_input = RandomWalker2D._dlap_df(lap_u, pu, gradout, num_grad, max_backprop)

        grad_input = randomwalker_tools.grad_fill(grad_input, seeds, 2).reshape(-1, seeds.shape[0], seeds.shape[1])
        grad_input = grad_input[None, ...]

        return torch.FloatTensor(grad_input).contiguous(), None, None, None

    @staticmethod
    def _dlap_df(lap_u, pu, gradout, num_grad, max_backprop):
        """
        Sampled back prop implementation
        grad_input: The gradient input for the previous layer
        """

        # Solver + sampling
        grad_input = np.zeros((2, pu.shape[0]))
        lap_u = coo_matrix(lap_u)
        ind_i, ind_j = lap_u.col, lap_u.row

        # mask n and w direction
        mask = (ind_j - ind_i) > 0
        ind_i, ind_j = ind_i[mask], ind_j[mask]

        # find the edge direction
        mask = ind_j - ind_i == 1
        dir_e = np.zeros_like(ind_i)
        dir_e[mask] = 1

        # Sampling
        if num_grad < np.unique(ind_i).shape[0]:
            u_ind = np.unique(ind_i)
            grad2do = np.random.choice(u_ind, size=num_grad, replace=False)
        else:
            grad2do = np.unique(ind_i)

        # Compute the choalesky decomposition
        ch_lap = cholesky(csc_matrix(lap_u))

        # find maxgrad for each region
        if max_backprop:
            c_max = np.argmax(np.abs(gradout), axis=1)
        else:
            # only biggest 10
            c_max = np.argsort(np.abs(gradout), axis=1)

        # Loops around all the edges
        for k, l, e in zip(ind_i, ind_j, dir_e):
            if k in grad2do:
                grad_input[e, k] = RandomWalker2D._compute_grad(pu, c_max, gradout, ch_lap, k, l)

        return grad_input

    @staticmethod
    def _compute_grad(pu, c_max, gradout, ch_lap, k, l):
        """
        k, l: pixel indices, referred to the unseeded laplacian
        ch_lap: choaleshy decomposition of the undseeded laplacian
        pu: unseeded output probability
        gradout: previous layer jacobian
        return: grad for the edge k, l
        """
        dl = np.zeros_like(pu)
        dl[l] = pu[k] - pu[l]
        dl[k] = pu[l] - pu[k]

        partial_grad = ch_lap.solve_A(dl[:, c_max[k]])
        grad = np.sum(gradout[:, c_max[k]] * partial_grad)
        return grad
