#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import utils


# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):
    # TODO: nstep LSTM
    def __init__(self, n_vocab, n_units):
        super(RNNForLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.LSTM(n_units, n_units)
            self.l2 = L.LSTM(n_units, n_units)
            self.l3 = L.Linear(n_units, n_vocab)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.loss = 0.

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0))
        h2 = self.l2(F.dropout(h1))
        y = self.l3(F.dropout(h2))
        return y

    def add_loss(self, y, t, normalize=None):
        loss = F.softmax_cross_entropy(y, t, normalize=False, reduce='mean')
        if normalize is not None:
            loss *= 1. * t.shape[0] / normalize
        self.loss += loss

    def add_batch_loss(self, ys, ts):
        y = F.concat(ys, axis=0)
        t = F.concat(ts, axis=0)
        batchsize = ts[0].shape[0]
        self.add_loss(y, t, normalize=batchsize)

    def pop_loss(self):
        loss = self.loss
        self.loss = 0.
        return loss
