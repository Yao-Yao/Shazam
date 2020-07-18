import numpy as np
from pprint import pprint as pp

class DisjointSet(object):
    def __init__(self):
        self.leader = {} # maps a member to the group's leader
        self.group = {} # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])

a = np.genfromtxt('result.csv', delimiter=',')
print 'result:'
print a.shape
print a

ds = DisjointSet()
s=set()
with np.errstate(invalid='ignore'):
    np.set_printoptions(threshold=10000)
    x, y = np.where(a > 0.8)
for i, px in enumerate(x):
    py = y[i]
    if py != px:
        print px, py, a[px][py]
        ds.add(px, py)
    if py == px:
        s.add(px)
pp(ds.leader)
pp(ds.group)

print 'nan: ', set(range(0,1775)) - s
