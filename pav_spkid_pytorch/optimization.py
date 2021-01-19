# Imports
import random
import numpy
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from utils import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit

# Steps for implementation:
# Step 1: A swarm of bees are randomly initialized in the search
#   space. Each solution is defined by a vector, x, where x = [Rs Rsh
#   Iph Isd1 Isd2 n1 n2] in the double diode model and x = [Rs Rsh Iph
#   Isd n] in the single diode model.
# Step 2: The value of the objective function for each bee is computed
#   based in Eq. (6).
# Step 3: The bees are ranked based on their objective functions.
# Step 4: Onlooker and scout bees are specified.
# Step 5: The position of the onlooker and scout bees is updated
#   according to their patterns.
# Step 6: If a bee exceeds the search space, it is replaced with the
#   previous position.
# Step 7: Steps 2 to 6 are repeated until itermax is met.
# Step 8: The best achievement of the swarm is selected as the
#   optimal solution.


# ----- ABSO ----- #
n = 2                   # Number of bees
itermax = 5000          # Number of times the algorithm is iterated
scouts = 0.15           # Percentage of scouts (of the total bees)
elite = 0.2             # Percentage of elite bees (of the onlookers)

d = 5                   # ORDER:: lr, hsize, epoch, momentum
upper_bound = [0]*5
lower_bound = [0]*5
upper_bound[0] = 0.2            # lr upper bpund
upper_bound[1] = 2000           # hsize upper bpund
upper_bound[2] = 100            # epoch upper bpund
upper_bound[3] = 100            # in_frames upper bpund
upper_bound[4] = 10000          # batch_size upper bpund

lower_bound[0] = 0.000001       # lr lower bpund
lower_bound[1] = 20             # hsize lower bpund
lower_bound[2] = 20             # epoch lower bpund
lower_bound[3] = 10             # in_frames lower bpund
lower_bound[4] = 50             # batch_size lower bpund

t_size = 2              # Tournament size for elite choosing

dmax_walk = 0.2
dmin_walk = 0.02

wb_max = we_max = 2.5
wb_min = we_min = 1.25

db_path = "work/mfcc"
ext = "mfcc"
spk2idx = "lists/spk2idx.json"
tr_list_file = "lists/class/all.train"
va_list_file = "lists/class/all.test"
save_path = "work/mcp"

# --- #

class Bee:
    "Bzzzzz"

    def __init__(self, beeNumber):
        self.bee_number = beeNumber
        self.bee_type = "none"
        self.bee_position = [0]*d
        self.bee_value = 0
        # Only for onlookers
        self.elite = 0
        self.best_achievement = [0]*d
    
    # - Getters - #
    def get_number(self):
        return self.bee_number
    def get_type(self):
        return self.bee_type
    def get_position(self):
        return self.bee_position
    def get_value(self):
        return self.bee_value
    def get_elite(self):
        return self.elite
    def get_achievement(self):
        return self.best_achievement

    # - Setters - #
    def set_type(self, type):
        self.bee_type = type
    def set_value(self, value):
        self.bee_value = value
    def set_elite(self, elite_number):
        self.elite = elite_number
    
    # - Set Position - #
        # scouts
    def inital_position(self, dim, upper_bound, lower_bound):
        for i in range(dim):
            alpha = random.random()
            self.bee_position[i] = lower_bound[i] + alpha*(upper_bound[i] - lower_bound[i])
    def walk_radius(self, dmax_walk, dmin_walk, iter, itermax):
        return (dmax_walk - (dmax_walk - dmin_walk)*(iter/itermax))
    def scoutUpdatePosition(self, dim, tau, upper_bound, lower_bound):
        r = random.uniform(-1, 1)
        wf = [0]*dim
        aux_pos = [0]*dim
        for i in range(dim):
            wf[i] = tau*(upper_bound[i] - lower_bound[i])
            aux_pos[i] = self.bee_position[i] + r*wf[i]
            if not out_of_bounds(dim, aux_pos, upper_bound, lower_bound) :
                self.bee_position[i] = aux_pos[i]
            else:
                pass
        # onlookers
    def converging_wb(self, wb_max, wb_min, iter, itermax):
        return (wb_max - (wb_max - wb_min)*(iter/itermax))
    def converging_we(self, we_max, we_min, iter, itermax):
        return (we_max - (we_max - we_min)*(iter/itermax))
    def onlookerUpdatePosition(self, dim, elite, upper_bound, lower_bound, wb, we):
        rb = random.random()
        re = random.random()
        aux_pos = [0]*dim
        for i in range(dim):
            aux_pos[i] = self.bee_position + wb*rb*(self.best_achievement[i] - self.bee_position[i]) + we*re*(elite.bee_position[i] - self.bee_position[i])
            if not out_of_bounds(dim, aux_pos, upper_bound, lower_bound) :
                self.bee_position[i] = aux_pos[i]
            else:
                pass

#  --------------- #
#  --------------- #
#  ---- TRAIN ---- #
#  --------------- #
#  --------------- #

def compute_accuracy(y_, y):
    pred = y_.max(1, keepdim=True)[1] 
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / y_.size(0)

def train_spkid_epoch(dloader, model, opt, 
                      epoch, log_freq):
    # setup train mode
    model.train()
    timings = []
    losses = []
    accs = []
    beg_t = timeit.default_timer()
    for bidx, batch in enumerate(dloader, start=1):
        X, Y = batch
        X = Variable(X)
        Y = Variable(Y)
        # reset any previous gradients in optimizer
        opt.zero_grad()
        # (1) Forward data through neural network
        Y_ = model(X)
        # (2) Compute loss (quantify mistake to correct towards giving good Y)
        # Loss is Negative Log-Likelihood, to reduce probability mismatch
        # between network output distribution and true distribution Y
        loss = F.nll_loss(Y_, Y)
        # (3) Backprop gradients
        loss.backward()
        # (4) Apply update to model parameters with optimizer
        opt.step()
        # Compute accuracy to check its increment during training
        acc = compute_accuracy(Y_, Y)
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()

        if bidx % log_freq == 0 or bidx >= len(dloader):
            print('TRAINING: {}/{} (Epoch {}) loss: {:.4f} acc:{:.2f} '
                  'mean_btime: {:.3f} s'.format(bidx, len(dloader), 
                                                epoch, loss.item(),
                                                acc,
                                                np.mean(timings)))
            losses.append(loss.item())
            accs.append(acc)
    return losses, accs

def eval_spkid_epoch(dloader, model, epoch, log_freq):
    # setup eval mode
    model.eval()
    va_losses = []
    va_accs = []
    timings = []
    beg_t = timeit.default_timer()
    for bidx, batch in enumerate(dloader, start=1):
        X, Y = batch
        X = Variable(X, volatile=True, requires_grad=False)
        Y = Variable(Y, volatile=True, requires_grad=False)
        Y_ = model(X)
        loss = F.nll_loss(Y_, Y)
        acc = compute_accuracy(Y_, Y)
        va_losses.append(loss.item())
        va_accs.append(acc)
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        beg_t = timeit.default_timer()
        if bidx % log_freq == 0 or bidx >= len(dloader):
            print('EVAL: {}/{} (Epoch {}) m_loss(so_far): {:.4f} mean_btime: {:.3f}'
                  's'.format(bidx, len(dloader), epoch, np.mean(va_losses),
                             np.mean(timings)))
    m_va_loss = np.mean(va_losses)
    m_va_acc = np.mean(va_accs)
    print('EVAL RESULT Epoch {} >> m_loss: {:.3f} m_acc: {:.2f}'
          ''.format(epoch, m_va_loss, m_va_acc))
    return [m_va_loss], [m_va_acc]

def train(lrate, hsize, n_epoch, in_frames, batch_size):
    max_acc = 0
    patience = 10
    log_freq = 100

    dset = SpkDataset(db_path, tr_list_file,
                      ext, spk2idx,
                      in_frames=in_frames)
    dloader = DataLoader(dset, batch_size=batch_size,
                         num_workers=1, shuffle=True, 
                         pin_memory=False)

    va_dset = SpkDataset(db_path, va_list_file,
                         ext, spk2idx,
                         in_frames=in_frames)
    va_dloader = DataLoader(va_dset, batch_size=batch_size,
                            num_workers=1, shuffle=True, 
                            pin_memory=False)
    input_dim = dset.input_dim
    num_spks = dset.num_spks
    # Cuda config
    # device = torch.cuda.device("cuda" if torch.cuda.is_available() else "cpu")
    # Feed Forward Neural Network
    model = nn.Sequential(nn.Linear(dset.input_dim * dset.in_frames, hsize),
                          nn.ReLU(),
                          nn.Linear(hsize, hsize),
                          nn.ReLU(),
                          nn.Linear(hsize, hsize),
                          nn.ReLU(),
                          nn.Linear(hsize, dset.num_spks),
                          nn.LogSoftmax(dim=1))


    #opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    opt = optim.Adam(model.parameters(), lr=lrate)
    tr_loss = []
    tr_acc = []
    va_loss = []
    va_acc = []
    best_val = np.inf
    # patience factor to validate data and get out of train earlier
    # of things do not improve in the held out dataset
    for epoch in range(n_epoch):
        tr_loss_, tr_acc_ = train_spkid_epoch(dloader, model, 
                                              opt, epoch,
                                              log_freq)
        va_loss_, va_acc_ = eval_spkid_epoch(va_dloader, model, 
                                             epoch, log_freq)
        if va_acc_[0] > max_acc :
            max_acc = va_acc_[0]
        if best_val <= va_loss_[0]:
            patience -= 1
            if patience <= 0:
                break
            mname = os.path.join(save_path,
                                 'e{}_weights.ckpt'.format(epoch))
        else:
            # reset patience
            best_val = va_loss_[0]
            patience = patience
            mname = os.path.join(save_path,
                                 'bestval_e{}_weights.ckpt'.format(epoch))
    return max_acc

#  --------------- #
#  --------------- #
#  --------------- #
#  --------------- #

def out_of_bounds(dim, pos, upper, lower):
    for i in range(dim):
        if pos[i] < upper[i] or pos[i] > lower[i] :
            return False
        else:
            return True

def tournament_selection(t_size, bee_list):
    participants = []
    for i in range(len(bee_list)):
        if bee_list[i].bee_type == "e" :
            participants.append(Bee(bee_list[i].get_number))
            participants[i].set_value = bee_list[i].bee_value*numpy.random.uniform(1, t_size, size=None)
    participants.sort(key=lambda b: b.bee_value, reverse=True)
    return participants[0]

def ABSO(n, d, scout, elite, upper_bound, lower_bound, dmax_walk, dmin_walk, wb_max, wb_min, we_max, we_min, itermax):
    # Step 1: Initialization of bees
    beeList = []
    for i in range(n):
        beeList.append(Bee(i))
        beeList[i].inital_position(d, upper_bound, lower_bound)
    # --- #

    # Step 7: Iteration of the 2 to 6 steps
    for iter in range(itermax):
        # Step 2: Compute our main algorithm for each bee
        for i in range(n):
            val = train(beeList[i].bee_position[0], int(beeList[i].bee_position[1]), int(beeList[i].bee_position[2]),
                        int(beeList[i].bee_position[3]), int(beeList[i].bee_position[4]))
            beeList[i].set_value(val)

        # Step 3: Rank th bees based on the result value
        beeList.sort(key=lambda b: b.bee_value, reverse=True)

        # Step 4: Specify type of bees
        s = n - scouts*n
        e = elite*n
        for i in range(n):
            if beeList[i].get_number > s :
                beeList[i].set_type("s")
            elif beeList[i].get_number < e :
                beeList[i].set_type("e")
            else:
                beeList[i].set_type("o")

        # Step 5: Update position
        # Step 6: Exceeding space bees are out or replaced
        for i in range(d):
            if beeList[i].bee_type == "s" :
                tau = beeList[i].walk_radius(dmax_walk, dmin_walk, iter, itermax)
                beeList[i].scoutUpdatePosition(d, tau, upper_bound, lower_bound)
            elif beeList[i].bee_type == "o":
                wb = beeList[i].converging_wb(wb_max, wb_min, iter, itermax)
                we = beeList[i].converging_we(we_max, we_min, iter, itermax)
                elite = tournament_selection(t_size, beeList)
                beeList[i].onlookerUpdatePosition(d, elite, upper_bound, lower_bound, wb, we)
            else:
                pass

    # Step 8: Final result
    beeList.sort(key=lambda b: b.bee_value, reverse=True)
    return beeList[0].bee_position

print("Starto!")
final_values = [0]*5
final_values = ABSO(n, d, scouts, elite, upper_bound, lower_bound, dmax_walk, dmin_walk, wb_max, wb_min, we_max, we_min, itermax)
print("Finish!!!\n -- lr = ", final_values[0], "\n -- hsize =", final_values[1], "\n -- epoch = ", final_values[2], "\n -- frames = ", final_values[3, "\n -- batch = ", final_values[4]])