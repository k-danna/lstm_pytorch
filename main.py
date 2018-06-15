#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from model import Model
from dataset import Dataset


def main():

    #currently available datasets
    data_files = [
        'shakespeare.txt',  #1.1 mb
        'grimms.txt',       #541 kb
        'bible.txt',        #4.4 mb
        'mobydick.txt',     #1.2 mb
        'tweets.txt',       #3.8 mb
        'tweets_clean.txt', #3.6 mb
    ]
    data_files = ['./data/%s' % f for f in data_files] #add dir loc

    #get data
    _set = data_files[1]
    data = Dataset(_set)
    print('[*] read all data from %s' % _set)

    #init model
    n = 256
    model = Model(data.x_shape, data.n_classes, n=256, recover=True)
    print('[*] created model')

    #train
    steps = 100000
    print('[*] training for %s steps' % steps)

    ##FIXME: open a loss window to view progress
        #https://stackoverflow.com/questions/5419888/reading-from-a-frequently-updated-file#5420116
        #https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
        #update the window at print intervals

    l_fname = 'losses.bin'
    l_file = open(l_fname, 'ab')

    h_tup = model.init_tup
    losses = []
    for i in range(steps):
        x, y = data.next_batch(32)
        x = [i.reshape(i.shape + (1,)) for i in x]
        loss, h_tup = model.train(x, y, h_tup)
        losses.append(loss)
        if (i+1) % (steps / 10) == 0:
            print('%s: %s' % (i+1, loss))
            #sample output
            s = model.sample(x[0], h_tup, 60)
            s = ''.join([data.enc_to_char[x] for x in s])
            print('   ###\n%s\n   ###' % s)
            #save model
            model.save()
            #dump losses to file and clear
            np.savetxt(l_file, np.asarray(losses))
            losses = []
    l_file.close()

    #test
    x, y = data.next_batch(32)
    x = [i.reshape(i.shape + (1,)) for i in x]
    s = model.sample(x[0], h_tup, 200)
    s = ''.join([data.enc_to_char[x] for x in s])
    print('#########################\n%s\n#########################' % s)

    visualize_losses(l_fname)

def visualize_losses(fname):
    #plot original losses
    losses = np.loadtxt(fname)
    plt.plot([_ for _ in range(len(losses))], losses, 'b', alpha=0.2)
    #smooth losses
    w = 0.9
    last = losses[0]
    for i in range(len(losses)):
        losses[i] = last * w + (1 - w) * losses[i]
        last = losses[i]

    #plot smoothed losses
    plt.plot([_ for _ in range(len(losses))], losses, 'b')
    plt.show()

if __name__ == '__main__':
    main()

