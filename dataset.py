
import numpy as np

#unsupervised dataset wrapper for unsupervised learning
class Dataset():
    def __init__(self, file_name):

        self._termin = '\n'

        self.data = [x for x in open(file_name, 'r').read()]

        #FIXME: just use ascii --> dec
            #set safe range and ignore other chars in input
            #ord('c') = 45, chr(45) = 'c'
        #set is unordered, labels will be non-deterministic if not sorted
        chars = sorted(set(self.data + [self._termin]))
        self.char_to_enc = {c:i for i,c in enumerate(chars)}
        self.enc_to_char = {i:c for i,c in enumerate(chars)}

        #useful info
        self.n_examples = len(self.data)
        self.n_classes = len(chars)
        self.x_shape = (self.n_classes,) #shape of a one-hot encoded x
        
        #internal use
        self._offset = 0
        self._max_offset = self.n_examples - 1

    def to_oh(self, i):
        oh = np.zeros(self.n_classes)
        oh[i] = 1
        return oh

    def from_oh(self, oh):
        return np.argmax(oh)

    def next_batch(self, n):
        #returns the next n examples, labels; wraps at end of set
        idx = min(self._max_offset, self._offset + n)
        batch_x = self.data[self._offset:idx]

        if idx == self._max_offset:
            batch_y = self.data[self._offset+1:idx] + [self._termin]
        else:
            batch_y = self.data[self._offset+1:idx+1]

        #FIXME: read a buffer in at a time instead of whole file
        #print('    collected: %s to %s' % (self._offset, idx))

        #calc how many to fetch before updating offset
        self._offset += n
        need = max(0, self._offset - self._max_offset)
        if idx == self._max_offset:
            self._offset = 0
        
        #encode to one hot
        batch_x = [self.to_oh(self.char_to_enc[c]) for c in batch_x]
        batch_y = [self.char_to_enc[c] for c in batch_y]

        #rnn input needs to be reshaped
        batch_x = np.asarray(batch_x)
        batch_x.reshape(batch_x.shape + (1,))

        #wrap around the dataset to make a full batch
        if need:
            t_x, t_y = self.next_batch(need)
            batch_x = np.append(batch_x, t_x, axis=0)
            batch_y = np.append(batch_y, t_y, axis=0)
        
        return batch_x, batch_y







