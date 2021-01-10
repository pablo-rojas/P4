# Imports
import random
import numpy
import math

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
n = 500                 # Number of bees
itermax = 5000          # Number of times the algorithm is iterated
scouts = 0.15           # Percentage of scouts (of the total bees)
elite = 0.2             # Percentage of elite bees (of the onlookers)

d = 4                   # ORDER:: lr, hsize, epoch, momentum
upper_bound = []
lower_bound = []
upper_bound[0] = 1      # lr upper bpund
upper_bound[1] = 2000   # hsize upper bpund
upper_bound[2] = 100    # epoch upper bpund
upper_bound[3] = 5      # momentum upper bpund

lower_bound[0] = 0      # lr lower bpund
lower_bound[1] = 50     # hsize lower bpund
lower_bound[2] = 40     # epoch lower bpund
lower_bound[3] = 0      # momentum lower bpund

t_size = 2              # Tournament size for elite choosing

dmax_walk = 0.2
dmin_walk = 0.02

wb_max = we_max = 2.5
wb_min = we_min = 1.25
# --- #

class Bee:
    "Bzzzzz"

    def __init__(self, beeNumber):
        self.bee_number = beeNumber
        self.bee_type = "none"
        self.bee_position = []
        self.bee_value = 0
        # Only for onlookers
        self.elite = 0
        self.best_achievement = []
    
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
        wf = []
        aux_pos = []
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
    def onlookerUpdatePosition(self, dim, elite_number, upper_bound, lower_bound, list, wb, we):
        rb = random.random()
        re = random.random()
        aux_pos = []
        for i in range(dim):
            aux_pos[i] = self.bee_position + wb*rb*(self.best_achievement[i] - self.bee_position[i]) + we*re*(list[elite_number].bee_position[i] - self.bee_position[i])
            if not out_of_bounds(dim, aux_pos, upper_bound, lower_bound) :
                self.bee_position[i] = aux_pos[i]
            else:
                pass



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
            participants[i] = Bee(bee_list[i].get_number)
            participants[i].set_value = bee_list[i].bee_value*numpy.random.uniform(1, t_size, size=None)
    participants.sort(key=lambda b: b.bee_value, reverse=True)
    return participants[0]

def ABSO(n, d, scout, elite, upper_bound, lower_bound, dmax_walk, dmin_walk, wb_max, wb_min, we_max, we_min, itermax):
    # Step 1: Initialization of bees
    beeList = []
    for i in range(n):
        beeList[i] = Bee(i)
        beeList[i].inital_position(d, upper_bound, lower_bound)
    # --- #

    # Step 7: Iteration of the 2 to 6 steps
    for iter in range(itermax):
        # Step 2: Compute our main algorithm for each bee
        for i in range(n):
            val = 0
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
            elif beeList[i].bee_type == "o" or beeList[i].bee_type == "e" : # Something about distance
                wb = beeList[i].converging_wb(wb_max, wb_min, iter, itermax)
                we = beeList[i].converging_we(we_max, we_min, iter, itermax)
                elite = tournament_selection(t_size, beeList)
                beeList[i].onlookerUpdatePosition(d, elite.get_number, upper_bound, lower_bound, beeList, wb, we)
            else:
                pass

    # Step 8: Final result
    beeList.sort(key=lambda b: b.bee_value, reverse=True)
    return beeList[0].bee_position
