# RNN Language Model by Chainer

This is a fast implementation of an RNN language model (RNNLM) by Chainer.
This repository is derived from the [Chainer example for RNNLM in PTB](https://github.com/chainer/chainer/tree/master/examples/ptb).

The network architecture is almost same as the "Medium" model in the paper, ["Recurrent Neural Network Regularization"](https://arxiv.org/pdf/1409.2329.pdf) by Wojciech Zaremba, Ilya Sutskever and Oriol Vinyals.
You can train an RNNLM in 1 miniute per epoch, with backprop length of 35 and batchsize of 20.

# How to Run

```
python -u train.py -g 0
```