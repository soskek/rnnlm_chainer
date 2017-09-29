from __future__ import print_function
import argparse
import copy
import numpy as np
import time

import chainer
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers

import utils
import nets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    args = parser.parse_args()

    def evaluate(model, iter):
        # Evaluation routine to be used for validation and test.
        evaluator = model.copy()  # to use different state
        evaluator.reset_state()  # initialize state
        sum_perp = 0
        data_count = 0
        outs = []
        targets = []
        model.loss = 0
        one_pack = args.batchsize * args.bproplen
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            for batch in copy.copy(iter):
                x, t = convert.concat_examples(batch, args.gpu)
                y = evaluator(x)
                outs.append(y)
                targets.append(t)
                data_count += 1
                if len(outs) >= one_pack:
                    evaluator.add_batch_loss(outs, targets)
                    sum_perp += evaluator.pop_loss().data
                    outs = []
                    targets = []
            if outs:
                evaluator.add_batch_loss(outs, targets)
                sum_perp += evaluator.pop_loss().data
        return np.exp(float(sum_perp) / data_count)

    # Load the Penn Tree Bank long word sequence dataset
    train, val, test = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    # Create the dataset iterators
    train_iter = utils.ParallelSequentialIterator(train, args.batchsize)
    val_iter = utils.ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = utils.ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    model = nets.RNNForLM(n_vocab, args.unit)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    sum_perp = 0
    count = 0
    iteration = 0
    is_new_epoch = 0
    best_val_perp = 1000000.
    start = time.time()
    print('Training start')
    while train_iter.epoch < args.epoch:
        iteration += 1
        outs = []
        targets = []
        model.loss = 0
        for i in range(args.bproplen):
            batch = train_iter.__next__()
            is_new_epoch += train_iter.is_new_epoch
            x, t = convert.concat_examples(batch, args.gpu)
            y = model(x)
            outs.append(y)
            targets.append(t)
            count += 1
        model.add_batch_loss(outs, targets)
        outs = []
        targets = []
        loss = model.pop_loss()
        sum_perp += loss.data
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters
        del loss

        if iteration % 500 == 0:
            time_str = time.strftime('%Y-%m-%d %H-%M-%S')
            mean_speed = (count // args.bproplen) / (time.time() - start)
            print('\ti {:}\tperp {:.3f}\t\t| TIME {:.3f}i/s ({})'.format(
                iteration, np.exp(float(sum_perp) / count), mean_speed, time_str))
            sum_perp = 0
            count = 0
            start = time.time()

        if is_new_epoch:
            tmp = time.time()
            val_perp = evaluate(model, val_iter)
            print('Epoch {:} val perp {:.3f}'.format(
                train_iter.epoch, val_perp))
            if val_perp < best_val_perp:
                  best_val_perp = val_perp
                  serializers.save_npz('best.model', model)
            start += (time.time() - tmp)
            optimizer.lr *= 0.85
            is_new_epoch = 0

    # Evaluate on test dataset
    print('test')
    print('load best model')
    serializers.load_npz('best.model', model)
    test_perp = evaluate(model, test_iter)
    print('test perplexity:', test_perp)


if __name__ == '__main__':
    main()
