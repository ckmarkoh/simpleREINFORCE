# -*- encoding:utf-8 -*-
import numpy as np

np.random.seed(1)


class Board(object):
    def __init__(self):
        self.HN = 7
        self.WN = 7
        self.AN = 4
        self.network = Network(self.HN * self.WN, self.AN)
        self.env = np.zeros((self.HN, self.WN))
        self.env[0, self.WN - 1] = 1.
        self.env[self.HN - 1, self.WN - 1] = -1.
        self.terminal = [[0, self.WN - 1], [self.HN - 1, self.WN - 1]]
        self.action_symbol = {0: u"↑", 1: u"↓", 2: u"←", 3: u"→"}
        self.terminal_symbol = {1: u"○", -1: u"×"}
        self.start = [self.HN / 2, 0]
        self.start_symbol = u"□"

    def check_boundary(self, x, y):
        if x < 0 or y < 0 or x > self.HN - 1 or y > self.WN - 1:
            return False
        else:
            return True

    def check_terminal(self, x, y):
        if [x, y] in self.terminal:
            return True
        else:
            return False

    def execute_action(self, action, x, y):
        if action == 0:
            return x - 1, y
        elif action == 1:
            return x + 1, y
        elif action == 2:
            return x, y - 1
        elif action == 3:
            return x, y + 1
        else:
            assert 0

    def encode_state(self, pos_x, pos_y):
        ipt = np.zeros((self.HN, self.WN))
        ipt[pos_x, pos_y] = 1
        return ipt.reshape(self.HN * self.WN, 1)

    def gen_action_str(self):
        s = u""
        for x in range(self.HN):
            for y in range(self.WN):
                if self.check_terminal(x, y):
                    s += self.terminal_symbol[self.env[x, y]]
                elif [x, y] == self.start:
                    s += self.start_symbol
                else:
                    state = self.encode_state(x, y)
                    action_prob = self.network.forward(state)
                    action = np.argmax(action_prob[:, 0])
                    s += self.action_symbol[action]
            s += u"\n"
        return s

    def run(self):
        iter = 1500
        max_seq = 40
        pid = 0
        for j in range(iter):
            [pos_x, pos_y] = self.start
            reward = 0
            state_seq = []
            action_prob_seq = []
            action_seq = []
            for i in range(max_seq):
                state = self.encode_state(pos_x, pos_y)
                action_prob = self.network.forward(state)
                action = np.argmax(np.random.multinomial(1, action_prob[:, 0]))
                npos_x, npos_y = self.execute_action(action, pos_x, pos_y)
                state_seq.append(state)
                action_prob_seq.append(action_prob)
                action_seq.append(action)
                if self.check_boundary(npos_x, npos_y):
                    pos_x, pos_y = npos_x, npos_y
                if self.check_terminal(pos_x, pos_y):
                    reward = self.env[pos_x, pos_y]
                    break
            if reward != 0:
                for idx in range(len(state_seq)):
                    state = state_seq[idx]
                    a_out = action_prob_seq[idx]
                    a_gold_idx = action_seq[idx]
                    a_gold = np.zeros((self.AN, 1))
                    a_gold[a_gold_idx, 0] = 1
                    self.network.backward(reward, state, a_gold, a_out)
            if j % (iter/10) == 0:
                print self.gen_action_str()
        print self.gen_action_str()




class Network(object):
    def __init__(self, s_ipt, s_opt):
        self.s_ipt = s_ipt
        self.s_opt = s_opt
        self.eta = 0.02
        self.W = np.random.rand(s_ipt, s_opt) / (s_ipt + s_opt)

    def forward(self, X):
        a_in = np.dot(X.T, self.W)
        a_out = (np.exp(a_in) / np.sum(np.exp(a_in))).T
        return a_out

    def backward(self, r, X, a_gold, a_out):
        self.W += self.eta * r * np.dot(X, (a_gold - a_out).T)


if __name__ == '__main__':
    b = Board()
    b.run()
