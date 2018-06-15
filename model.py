
import torch
import numpy as np
import pickle

class Model():
    def __init__(self, x_shape, n_classes, n=256, recover=False):
        np.random.seed(42)
        torch.manual_seed(42)

        #misc info
        self.x_shape = x_shape
        self.n_classes = n_classes

        #gradient steps taken
        self.t = 0

        self.use_lstm = True
        if self.use_lstm:
            #lstm cell weights
            self.w = {
                #input weights, bias
                'xi': self.init_var((n, n_classes)), #input to hidden
                'hi': self.init_var((n, n)), #hidden to hidden
                'bi': self.init_var((n, 1)), #hidden bias
                #forget weights, bias
                'xf': self.init_var((n, n_classes)), #input to hidden
                'hf': self.init_var((n, n)), #hidden to hidden
                'bf': self.init_var((n, 1)), #hidden bias
                #output weights, bias
                'xo': self.init_var((n, n_classes)), #input to hidden
                'ho': self.init_var((n, n)), #hidden to hidden
                'bo': self.init_var((n, 1)), #hidden bias
                #hidden state weights, bias
                'xc': self.init_var((n, n_classes)), #input to hidden
                'hc': self.init_var((n, n)), #hidden to hidden
                'bc': self.init_var((n, 1)), #hidden bias
                #classify weights, bias
                'hy': self.init_var((n_classes, n)), #hidden to output
                'by': self.init_var((n_classes, 1), 'zero'), #output bias
            }
        else:
            #vanilla tanh rnn cell weights
            self.w = {
                'xh': self.init_var((n, n_classes)), #input to hidden
                'hh': self.init_var((n, n)), #hidden to hidden
                'hy': self.init_var((n_classes, n)), #hidden to output
                'bh': self.init_var((n, 1), 'zero'), #hidden bias
                'by': self.init_var((n_classes, 1), 'zero'), #output bias
            }


        #for adam update
        self.m = {k: 0.0 for k in self.w}
        self.v = {k: 0.0 for k in self.w}

        #for init h_tup
        self.init_tup = (np.zeros((n, 1)), np.zeros((n, 1)))

        if recover:
            self.restore()

    def init_var(self, shape, dist='simple'):
        #xavier init weights, bias inits to zero
        c = 6.0 if dist == 'uniform' else 2.0
        in_out = shape[-2] + shape[-1]
        std = np.sqrt(c / in_out)
        if dist == 'uniform':
            var = np.random.uniform(shape, minval=-std, maxval=std)
        elif dist == 'normal':
            var = np.random.uniform(shape, mean=0.0, stddev=std)
        elif dist == 'simple':
            var = np.random.normal(0.0, 0.01, shape)
        elif dist == 'zero':
            var = np.zeros(shape, dtype=np.float32)
        return torch.tensor(var, dtype=torch.float32, 
                requires_grad=True)
        
    def vanilla_cell(self, x, h_tup):
        h_prev, _ = h_tup
        #weight x input
        x_t = torch.mm(self.w['xh'], x)
        #weight h input
        h_t = torch.mm(self.w['hh'], h_prev) + self.w['bh']
        #vanilla cell op
        h = torch.tanh(x_t + h_t)
        return h,h

        #return torch.tanh(torch.mm(self.w['xh'], x) + 
        #        torch.mm(self.w['hh'], h_prev) + self.w['bh'])

    def lstm_cell(self, x, h_tup, use_peephole=False):
        h_prev, c_prev = h_tup

        #lstm with forget gate, note unique per line w,u,b
            #'@' is hadamard product
        #f = sigmoid( w*x + u*h_prev + b )
        #i = sigmoid( w*x + u*h_prev + b )
        #o = sigmoid( w*x + u*h_prev + b )
        #c = f@c_prev + i@tanh( w*x + u*h_prev + b )
        #h = o@tanh( c )

        #lstm with forget gate, peephole connections
            #'@' is hadamard product
        #f = sigmoid( w*x + u*c_prev + b )
        #i = sigmoid( w*x + u*c_prev + b )
        #o = sigmoid( w*x + u*c_prev + b )
        #c = f@c_prev + i@tanh( w*x + u*c_prev + b )
        #h = o@c

        #configure peephole connections
        h_prev = c_prev if use_peephole else h_prev
        #activ_fn = lambda x: x if use_peephole else torch.tanh

        #forget gate
        x_f = torch.mm(self.w['xf'], x)
        h_f = torch.mm(self.w['hf'], h_prev) + self.w['bf']
        f = torch.sigmoid(x_f + h_f)
        #input gate
        x_i = torch.mm(self.w['xi'], x)
        h_i = torch.mm(self.w['hi'], h_prev) + self.w['bi']
        i = torch.sigmoid(x_i + h_i)
        #ouput gate
        x_o = torch.mm(self.w['xo'], x)
        h_o = torch.mm(self.w['ho'], h_prev) + self.w['bo']
        o = torch.sigmoid(x_o + h_o)
        #cell state
        x_c = torch.mm(self.w['xc'], x)
        h_c = torch.mm(self.w['hc'], h_prev) + self.w['bc']
        c = f * c_prev + i * torch.tanh(x_c + h_c)
        #output
        #h = o * activ_fn(c)
        h = o * torch.tanh(c)

        #print(torch.isnan(h), torch.isnan(c))
        

        return h,c

    def classify_logits(self, logits):
        #unnormalized log probs for next chars
        log_probs = torch.mm(self.w['hy'], logits) + self.w['by'] 
        #softmax --> probs for next chars
        probs = torch.exp(log_probs) / torch.sum(torch.exp(log_probs))
        return probs

    def forward(self, x, y, h_tup):
        #convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        h_tup = [torch.tensor(x, dtype=torch.float32) for x in h_tup]
        loss = torch.tensor(0.0, dtype=torch.float32)
        for i in range(len(x)):
            #hidden state, defaults to basic tanh cell
            if self.use_lstm:
                h_tup = self.lstm_cell(x[i], h_tup)
            else:
                h_tup = self.vanilla_cell(x[i], h_tup) 

            #to probs
            p = self.classify_logits(h_tup[0])

            #cross entropy loss
            loss += -torch.log(p[y[i], 0])

        return loss, h_tup
        
    def apply_grads(self, lr=1e-4):
        with torch.no_grad():
            self.t += 1
            b1, b2, e = 0.9, 0.999, 1e-8
            lr_t = lr * np.sqrt((1 - np.power(b2, self.t)) / 
                    (1 - np.power(b1, self.t)))
            for k in self.w:
                g = self.w[k].grad.data
                
                #adam update
                self.m[k] = b1 * self.m[k] + (1 - b1) * g
                self.v[k] = b2 * self.v[k] + (1 - b2) * torch.pow(g, 2)
                self.w[k].data -= lr_t * self.m[k] / (
                        torch.sqrt(self.v[k]) + e)
                
                #vanilla update
                #self.w[k].data -= lr * self.w[k].grad.data

                #zero out gradient
                self.w[k].grad.zero_()

    def train(self, x, y, h_tup): #assumes x, y are oh
        loss, h_tup = self.forward(x, y, h_tup)    
        loss.backward()
        self.apply_grads()
        #note that detach must happen before returning
        return loss.detach().numpy(), [x.detach().numpy() for x in h_tup]

    def sample(self, x, h_tup, n):
        with torch.no_grad():
            out = []
            x = torch.tensor(x, dtype=torch.float32)
            h_tup = [torch.tensor(x, dtype=torch.float32) for x in h_tup]
            for i in range(n):
                if self.use_lstm:
                    h_tup = self.lstm_cell(x, h_tup)
                else:
                    h_tup = self.vanilla_cell(x, h_tup) 
                p = self.classify_logits(h_tup[0])
                #idx = np.argmax(p)
                idx = np.random.choice(self.n_classes, 
                        p=p.detach().numpy().ravel())
                #convert to oh
                x = torch.zeros((self.n_classes, 1))
                x[idx] = 1

                out.append(idx)
        return out

    def to_oh(self, i):
        with torch.no_grad():
            oh = torch.zeros((self.n_classes, 1))
            oh[i] = 1
            return oh

    def save(self):
        #note that this overwrites the existing file
        with open('w.pkl', 'wb') as f:
            pickle.dump(self.t, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.w, f, protocol=pickle.HIGHEST_PROTOCOL)

    def restore(self):
        try:
            with open('w.pkl', 'rb') as f:
                self.t = pickle.load(f)
                w = pickle.load(f)
                assert(w.keys() == self.w.keys())
                for k in self.w:
                    assert(w[k].shape == self.w[k].shape)
                self.w = w
            print('[+] recovered weights at step %s' % self.t)
        except:
            print('[-] could not load weights')











