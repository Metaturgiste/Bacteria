# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 13:56:46 2021

@author: celia
"""

from math import *
import numpy as np
import matplotlib.pyplot as plt
import csv
from operator import mul
from functools import reduce
import sys
from os.path import exists

N_string = ['2', '5', '8', '11']
N_liste = [2, 5, 8, 11]
T_string = ['10', '100', '1000']
T_liste = [10, 100, 1000]
N = N_liste[1]
T = T_liste[1]
tau_liste = [[0.5, 0.1, 1], [0.1, 1, 3, 10, 0.01, 0.1, 1, 3, 10],
             [0.1, 0.5, 1, 3, 10, 30, 100, 0.01, 0.1, 0.5, 1, 3, 10, 30, 100]]
tau = [0]

E_liste = ['cos(x)', 'tanh', '0', '1', 'croissant', 'random', 'Par_palier', 'sin_amorti','cos_petit']
E = E_liste[0]
F = E_liste[0]

facteurRand = 1
nbTour = 100
nPlayer = 50
selectPop = 10
nbBad = 3
nbRandom = 0
nbMutation = 4
nbNew = 2
nbVib = 20
depthVib = 5
randE = []

M_List = ["M" + str(i) for i in range(N)]
data_list = ["cminus" + str(i) for i in range(N - 1)] + ["cplus" + str(i) for i in range(N - 1)] + ["tauminus" + str(i)
                                                                                                    for i in
                                                                                                    range(N - 1)] + [
                "taue" + str(i) for i in range(N)] + ["e" + str(i) for i in range(N)] + ["r" + str(i) for i in range(N)]
data_list_no_tau = ["cminus" + str(i) for i in range(N - 1)] + ["cplus" + str(i) for i in range(N - 1)] + ["e" + str(i)
                                                                                                           for i in
                                                                                                           range(N)] + [
                       "r" + str(i) for i in range(N)]
M = [[0] * T for i in range(N)]
X = [i for i in range(T)]
wf, we, wc, wr = 1, 0, 0, 0
meta_para_liste = [N_liste, T_liste, E_liste, E_liste]
len_meta_list = [len(i) for i in meta_para_liste]
full_len = reduce(mul, len_meta_list, 1)


def Env(x, Ef):
    if E == "Par_palier":
        if (x >= 0 and x < 6) or (x >= 30 and x < 47) or (x >= 70 and x < 79):
            return -1
        elif (x >= 6 and x < 11) or (x >= 19 and x < 30) or (x >= 47 and x < 51) or (x >= 60 and x < 70) or (
                x >= 79 and x < 87):
            return 0
        else:
            return 1
    elif E == "cos(x)":
        return cos(x / 12)
    elif E == "tanh":
        return tanh((10 * x / T) - 5)
    elif E == "0":
        return 0
    elif E == "1":
        return 1
    elif E == "random":
        return Ef[x]
    elif E == "sin_amorti":
        return ((x + 1) ** (-1 / 3)) * sin(12 * (x / T))
    elif E == 'croissant':
        return 2 * (x / T) - 1
    elif E == 'cos_petit':
        return 0.1*cos(x/12)


def noisei(x):
    return 0


def noiser(x):
    return 0


def m(deltat, tau):
    return (np.exp(-deltat / tau) / tau) * ((deltat / tau) - (1 / 2) * (deltat / tau) ** 2)


def gain(cminus, cplus, tauminus, taue, e, r):
    global M
    R = [0] * T
    Fit = [0] * T
    Ef = []
    if E == "random":
        Ef = [2 * np.random.random() - 1 for i in range(T)]
    M = [[Env(0, Ef)] * T for i in range(N)]
    for t in range(T - 1):
        for i in range(N):
            sume = [Env(s, Ef) * m(t - s, taue[i]) for s in range(t + 1)]
            if i == 0:
                M[0][t + 1] = tanh(cplus[0] * M[1][t] + e[0] * sum(sume) + noisei(t))
            else:
                sumi = [M[i - 1][s] * m(t - s, tauminus[i - 1]) for s in range(t + 1)]
                if i < N - 1:
                    M[i][t + 1] = tanh(
                        cminus[i - 1] * sum(sumi) + cplus[i] * M[i + 1][t] + e[i] * sum(sume) + noisei(t))
                else:
                    M[N - 1][t + 1] = tanh(cminus[N - 2] * sum(sumi) + e[N - 1] * sum(sume) + noisei(t))

    for t in range(T):
        sumr = [r[j] * M[j][t] for j in range(N)]
        R[t] = tanh(sum(sumr) + noiser(t))
        a = [elt ** 2 for elt in e]
        b = [elt ** 2 for elt in cminus]
        c = [elt ** 2 for elt in cplus]
        b = b + c
        d = [elt ** 2 for elt in r]
        Fit[t] = np.exp(- (wf * (R[t] - Env(t, Ef)) ** 2) - (we * sum(a)) - (wc * sum(b)) - (wr * sum(d)))
    return sum(Fit)


def gamma(L):
    """Takes a list of 6N-3 parameters and returns the value of the gain"""
    cminus = L[:(N - 1)]
    cplus = L[(N - 1):(2 * N - 2)]
    tauminus = L[(2 * N - 2):(3 * N - 3)]  # We suppose the values are already sorted
    taue = L[(3 * N - 3):(4 * N - 3)]  # We suppose the values are already sorted
    e = L[(4 * N - 3):(5 * N - 3)]
    r = L[(5 * N - 3):(6 * N - 3)]
    return gain(cminus, cplus, tauminus, taue, e, r)


def gamma_no_tau(L, tauminus, taue):
    """Takes a list of 4N-2 parameters and returns the value of the gain"""
    cminus = L[:(N - 1)]
    cplus = L[(N - 1):(2 * N - 2)]
    e = L[(2 * N - 2):(3 * N - 2)]
    r = L[(3 * N - 2):(4 * N - 2)]
    return gain(cminus, cplus, tauminus, taue, e, r)


def gain_draw(cminus, cplus, tauminus, taue, e, r, wf, we, wc, wr):
    global M
    R = [0] * T
    Fit = [0] * T
    Ef = []
    if E == "random":
        Ef = [2 * np.random.random() - 1 for j in range(T)]
    M = [[Env(0, Ef)] * T for j in range(N)]
    for t in range(T - 1):
        for i in range(N):
            sume = [Env(s, Ef) * m(t - s, taue[i]) for s in range(t + 1)]
            if i == 0:
                M[0][t + 1] = tanh(cplus[0] * M[1][t] + e[0] * sum(sume) + noisei(t))
            else:
                sumi = [M[i - 1][s] * m(t - s, tauminus[i - 1]) for s in range(t + 1)]
                if i < N - 1:
                    M[i][t + 1] = tanh(
                        cminus[i - 1] * sum(sumi) + cplus[i] * M[i + 1][t] + e[i] * sum(sume) + noisei(t))
                else:
                    M[N - 1][t + 1] = tanh(cminus[N - 2] * sum(sumi) + e[N - 1] * sum(sume) + noisei(t))

    for t in range(T):
        sumr = [r[j] * M[j][t] for j in range(N)]
        R[t] = tanh(sum(sumr) + noiser(t))
        a = [elt ** 2 for elt in e]
        b = [elt ** 2 for elt in cminus]
        c = [elt ** 2 for elt in cplus]
        b = b + c
        d = [elt ** 2 for elt in r]
        Fit[t] = np.exp(- (wf * (R[t] - Env(t, Ef)) ** 2) - (we * sum(a)) - (wc * sum(b)) - (wr * sum(d)))
    draw(R, Ef)
    return sum(Fit)


def draw(R, Ef):
    global X
    X = [i for i in range(T)]
    plt.figure()
    if E == "random":
        Xe = X
    else:
        Xe = [i / 100 for i in range(100 * T)]
    Ye = [Env(x, Ef) for x in Xe]
    plt.plot(Xe, Ye, X, R)
    plt.savefig("results_N_" + str(N) + "_T_" + str(T) + "_optimized_on_" + F + "_tested_on_" + E + ".png")
    plt.close()


def gamma_draw(L):
    """Takes a list of 6N-3 parameters and returns the value of the gain"""
    cminus = L[:(N - 1)]
    cplus = L[(N - 1):(2 * N - 2)]
    tauminus = L[(2 * N - 2):(3 * N - 3)]  # We suppose the values are already sorted
    taue = L[(3 * N - 3):(4 * N - 3)]  # We suppose the values are already sorted
    e = L[(4 * N - 3):(5 * N - 3)]
    r = L[(5 * N - 3):(6 * N - 3)]
    gain_opt = gain_draw(cminus, cplus, tauminus, taue, e, r, wf, we, wc, wr)
    file = open("results_N_" + str(N) + "_T_" + str(T) + "_optimized_on_" + F + "_tested_on_" + E + ".txt", 'w')
    file.write("Paramètres optimisés : " + str(L) + "\n")
    file.write("Gain obtenu : " + str(gain_opt) + "\n")
    file.write("Mémoires : " + str(M) + "\n")
    file.write("Variations de Mémoires : " + str(variation_mem(M)))
    file.close()


def gain_pred(cminus, cplus, tauminus, taue, e, r, wf, we, wc, wr):
    global M
    R = [0] * T
    Fit = [0] * T
    Ef = []
    if E == "random":
        Ef = [2 * np.random.random() - 1 for j in range(T)]
    R[0] = Env(0, Ef)
    M = [[Env(0, Ef)] * T for j in range(N)]
    for t in range(T - 1):
        for i in range(N):
            sume = [R[s] * m(t - s, taue[i]) for s in range(t + 1)]
            if i == 0:
                M[0][t + 1] = tanh(cplus[0] * M[1][t] + e[0] * sum(sume) + noisei(t))
            else:
                sumi = [M[i - 1][s] * m(t - s, tauminus[i - 1]) for s in range(t + 1)]
                if i < N - 1:
                    M[i][t + 1] = tanh(
                        cminus[i - 1] * sum(sumi) + cplus[i] * M[i + 1][t] + e[i] * sum(sume) + noisei(t))
                else:
                    M[N - 1][t + 1] = tanh(cminus[N - 2] * sum(sumi) + e[N - 1] * sum(sume) + noisei(t))

    for t in range(T):
        sumr = [r[j] * M[j][t] for j in range(N)]
        R[t] = tanh(sum(sumr) + noiser(t))
        a = [elt ** 2 for elt in e]
        b = [elt ** 2 for elt in cminus]
        c = [elt ** 2 for elt in cplus]
        b = b + c
        d = [elt ** 2 for elt in r]
        Fit[t] = np.exp(- (wf * (R[t] - Env(t, Ef)) ** 2) - (we * sum(a)) - (wc * sum(b)) - (wr * sum(d)))
    draw_pred(R, Ef)
    return sum(Fit)


def draw_pred(R, Ef):
    global X
    X = [i for i in range(T)]
    plt.figure()
    if E == "random":
        Xe = X
    else:
        Xe = [i / 100 for i in range(100 * T)]
    Ye = [Env(x, Ef) for x in Xe]
    plt.plot(Xe, Ye, X, R)
    plt.savefig("results_N_" + str(N) + "_T_" + str(T) + "_optimized_on_" + E + "_prediction.png")
    plt.close()


def prediction(L):
    cminus = L[:(N - 1)]
    cplus = L[(N - 1):(2 * N - 2)]
    tauminus = L[(2 * N - 2):(3 * N - 3)]  # We suppose the values are already sorted
    taue = L[(3 * N - 3):(4 * N - 3)]  # We suppose the values are already sorted
    e = L[(4 * N - 3):(5 * N - 3)]
    r = L[(5 * N - 3):(6 * N - 3)]
    gain_opt = gain_pred(cminus, cplus, tauminus, taue, e, r, wf, we, wc, wr)
    file = open("results_N_" + str(N) + "_T_" + str(T) + "_optimized_on_" + E + "_prediction.txt", 'w')
    file.write("Paramètres optimisés : " + str(L) + "\n")
    file.write("Gain obtenu : " + str(gain_opt) + "\n")
    file.write("Mémoires : " + str(M) + "\n")
    file.write("Variations de Mémoires : " + str(variation_mem(M)))
    file.close()


def parcours_local(param, fit, step):
    end = True
    for i in range(len(param)):
        prev_val = param[i]
        next_val_l = param[i]
        next_val_r = param[i]
        d = 1
        n_l = next_val_l - d * step
        fit_l = fit
        if i in range(2 * N - 1, 3 * N - 3) or i in range(3 * N - 2, 4 * N - 3):
            while d <= 1 and n_l > param[i - 1]:
                param[i] = n_l
                f_n = gamma(param)
                if f_n > fit_l:
                    next_val_l = n_l
                    d = 1
                    n_l = next_val_l - d * step
                    fit_l = f_n
                else:
                    d += 1
                    n_l = next_val_l - d * step
        else:
            while d <= 1 and n_l > 0:
                param[i] = n_l
                f_n = gamma(param)
                if f_n > fit_l:
                    next_val_l = n_l
                    d = 1
                    n_l = next_val_l - d * step
                    fit_l = f_n
                else:
                    d += 1
                    n_l = next_val_l - d * step
        d = 1
        n_r = next_val_r + d * step
        fit_r = fit
        if i in range(2 * N - 2, 3 * N - 4) or i in range(3 * N - 3, 4 * N - 4):
            while d <= 1 and n_r < param[i + 1]:
                param[i] = n_r
                f_n = gamma(param)
                if f_n > fit_r:
                    next_val_r = n_r
                    d = 1
                    n_r = next_val_r + d * step
                    fit_r = f_n
                else:
                    d += 1
                    n_r = next_val_r + d * step
        else:
            while d <= 1:
                param[i] = n_r
                f_n = gamma(param)
                if f_n > fit_r:
                    next_val_r = n_r
                    d = 1
                    n_r = next_val_r + d * step
                    fit_r = f_n
                else:
                    d += 1
                    n_r = next_val_r + d * step
        if next_val_l == prev_val:
            param[i] = next_val_r
        elif next_val_r == prev_val:
            param[i] = next_val_l
        elif fit_l > fit_r:
            end &= False
            param[i] = next_val_l
            fit = fit_l
        else:
            end &= False
            param[i] = next_val_r
            fit = fit_r
    return param, end, fit


def parcours_local_no_tau(param, fit, step, tauminus, taue):
    end = True
    for i in range(len(param)):
        prev_val = param[i]
        next_val_l = param[i]
        next_val_r = param[i]
        d = 1
        n_l = next_val_l - d * step
        fit_l = fit
        while d <= 1 and n_l > 0:
            param[i] = n_l
            f_n = gamma_no_tau(param, tauminus, taue)
            if f_n > fit_l:
                next_val_l = n_l
                d = 1
                n_l = next_val_l - d * step
                fit_l = f_n
            else:
                d += 1
                n_l = next_val_l - d * step
        d = 1
        n_r = next_val_r + d * step
        fit_r = fit
        while d <= 1:
            param[i] = n_r
            f_n = gamma_no_tau(param, tauminus, taue)
            if f_n > fit_r:
                next_val_r = n_r
                d = 1
                n_r = next_val_r + d * step
                fit_r = f_n
            else:
                d += 1
                n_r = next_val_r + d * step
        if next_val_l == prev_val:
            param[i] = next_val_r
        elif next_val_r == prev_val:
            param[i] = next_val_l
        elif fit_l > fit_r:
            end &= False
            param[i] = next_val_l
            fit = fit_l
        else:
            end &= False
            param[i] = next_val_r
            fit = fit_r
    return param, end, fit


facteurRand = 1


def rand_R():
    return (1 / (1 - np.random.random())) * facteurRand


def sort_tau(player):
    # If the tau have been specified, they are replaced by there specified value
    if (len(tau) == 2 * N - 1):
        player[2 * N - 2:3 * N - 3] = tau[:N - 1]
        player[3 * N - 3:4 * N - 3] = tau[N - 1:]
        # To sort the values of tauminus
        player[2 * N - 2:3 * N - 3] = sorted(player[2 * N - 2:3 * N - 3])
        # To sort the values of taue
        player[3 * N - 3:4 * N - 3] = sorted(player[3 * N - 3:4 * N - 3])
    return player


def gen_player():
    player = [0.1] * (6 * N - 3)

    # Fill the list with rand_R() generated values
    for i in range(6 * N - 3):
        player[i] = rand_R()

    sort_tau(player)
    return player


def gen_player_no_tau():
    player = [0.1] * (4 * N - 2)

    # Fill the list with rand_R() generated values
    for i in range(4 * N - 2):
        player[i] = rand_R()
    return player


def create_player():
    player = gen_player()
    while (gamma(player) == 0):
        player = gen_player()
    return player


def create_player_gen():
    player = gen_player()
    return player


def create_player_no_tau(tauminus, taue):
    player = gen_player_no_tau()
    while (gamma_no_tau(player, tauminus, taue) == 0):
        player = gen_player_no_tau()
    return player


def opti_parcours_local(param=None):
    if param:
        end = False
        step = 0.1
        fit = gamma(param)
        counter = 0
        ncount = 0
        while counter <= 2:  # We end once every parameter is locally optimized at the same time
            param, end, fit = parcours_local(param, fit, step)
            ncount+=1
            print(ncount, " turn done")
            if end:
                print("Division du pas")
                counter += 1
                step /= 8
        return param
    else:
        params = [create_player() for i in range(10)]
        c = 0
        opt_fit = 0
        f = open("opti_local.txt", 'w')
        for param in params:
            end = False
            step = 0.1
            fit = gamma(param)
            counter = 0
            while counter <= 3:  # We end once every parameter is locally optimized at the same time
                param, end, fit = parcours_local(param, fit, step)
                print("1 Tour")
                if end:
                    counter += 1
                    step /= 8
            print("Paramètres optimisés")
            c += 1
            f.write("Mémoires à la fin : " + str(M) + "\n")
            f.write("Paramètres optimisés : " + str(param) + "\n")
            f.write("Gain optimal : " + str(fit) + "\n")
            M_data = np.array(M)
            M_data = M_data.transpose()
            if fit > opt_fit:
                opt_param = param
                opt_fit = fit
        return opt_param


def opti_parcours_local_no_tau():
    tauminus = [T / (10 ** (N - 1 - j)) for j in range(N - 1)]
    taue = [T / (10 ** (N - j)) for j in range(N)]
    opt_fit = 0
    params = [create_player_no_tau(tauminus, taue) for i in range(1)]
    for param in params:
        end = False
        step = 0.1
        fit = gamma_no_tau(param, tauminus, taue)
        counter = 0
        while counter <= 0:  # We end once every parameter is locally optimized at the same time
            param, end, fit = parcours_local_no_tau(param, fit, step, tauminus, taue)
            if end:
                counter += 1
                step /= 8
        print("Paramètres optimisés")
        if fit > opt_fit:
            opt_param = param
            opt_fit = fit
    return opt_param


def rate_player(player):
    sort_tau(player)
    return gamma(player)


def rate_pop(pop):
    nbPlayer = len(pop)
    rating = [0] * (nbPlayer)
    for i in range(nbPlayer):
        rating[i] = rate_player(pop[i])
    return rating


def sort_pop(pop):
    rating = rate_pop(pop)
    tuples = sorted(zip(rating, pop), reverse=True)
    rating, pop = [t[0] for t in tuples], [t[1] for t in tuples]
    # print('These are the ratings :')
    # print(rating)
    return pop


def rand_player(pop):
    nbPlayer = len(pop)
    return floor(np.random.random() * nbPlayer)


def rand_list():
    lenght = 6 * N - 3
    return floor(np.random.random() * lenght)


def mutation(pop):
    for i in range(nbMutation):
        pop[rand_player(pop)][rand_list()] = rand_R()
    return pop


def new_player(pop):
    for i in range(nbNew):
        pop[selectPop + nbBad + nbRandom + i] = gen_player()
    return pop


def croisement(pop):
    for i in range(selectPop + nbBad + nbRandom + nbNew, nPlayer):
        cut = rand_list()
        father = rand_player(pop)
        mother = rand_player(pop)
        while father == mother:
            father = rand_player(pop)
            mother = rand_player(pop)
        son = pop[father][:cut] + pop[mother][cut:]
        pop[i] = son.copy()
    return pop


def select_pop(pop):
    pop = sort_pop(pop).copy()
    for i in range(nbBad):
        pop[selectPop + i] = pop[-i].copy()
    for i in range(nbRandom):
        pop[selectPop + nbBad + i] = pop[rand_list()].copy()
    return pop


def init_pop(nbPlayer):
    pop = [0.1] * nbPlayer
    for i in range(nbPlayer):
        pop[i] = create_player_gen()
    return pop


def vibration(pop):
    for i in range(nbVib):
        ind = rand_player(pop)
        p = rand_list()
        d = - depthVib * np.random.random() - 2
        a = np.random.random() - 0.5
        pop[ind][p] += a * 10 ** d
    return (pop)


def genetic_opt():
    pop = init_pop(nPlayer)
    for i in range(nbTour):
        print('This is round number: ' + str(i))
        pop = select_pop(pop)
        # print('The best valuation being :')
        # print(gamma(pop[0])[0])
        new_player(pop)
        croisement(pop)
        mutation(pop)
        vibration(pop)
    sort_pop(pop)
    return pop[0]


def variation_mem(M):
    resu = []

    for mem in range(len(M)):
        resu.append(0)
        moy = sum([a for a in M[mem]]) / len(M[mem])
        for i in M[mem]:
            resu[mem] += (moy - i) ** 2
    return resu


def select_meta():
    global E, N, T
    for Ni in range(len(N_liste)):
        N = N_liste[Ni]
        for Ti in range(len(T_liste)):
            T = T_liste[Ti]
            for Ei in range(len(E_liste)):
                E = E_liste[Ei]
                yield Ni, Ti, Ei


def res_init(res=[], i=0):
    for j in range(len(meta_para_liste[i])):
        res.append([])
        if i < len(meta_para_liste) - 1:
            res[-1] = res_init(res[-1], i + 1)
    return res


current_position = 0

NT_long = 0
for i in N_liste:
    for j in T_liste:
        NT_long += i * j * (len(E_liste) ** 2)


def how_long():
    global current_position
    current_position += N * T
    print('\r', floor(current_position / NT_long * 10000) / 100, '%', sep=' ', end=' ', flush=True)
    return


def benchmark():
    global E, F
    res = res_init()
    for (Ni, Ti, Ei) in select_meta():
        how_long()
        P_opt = genetic_opt()  # opti_parcours_local()
        for Fi in range(len(E_liste)):
            F = E_liste[Fi]
            F, E = E, F
            res_gamma = gamma_draw(P_opt)
            res[Ni][Ti][Fi][Ei].append(P_opt)  # Paramètres
            res[Ni][Ti][Fi][Ei].append(res_gamma)  # Gain
            res[Ni][Ti][Fi][Ei].append(M)  # Mémoires
            res[Ni][Ti][Fi][Ei].append(variation_mem(M))  # Variations de mémoires
            F, E = E, F

    return res

def no_repeat(n, t, e, f):
    file_exists = exists("results_N_" + str(N) + "_T_" + str(T) + "_optimized_on_" + F + "_tested_on_" + E + ".txt")
    return file_exists

def main(i, j, k):
    global N, T, E, F
    N = N_liste[i]
    T = T_liste[j]
    E = E_liste[k]

    print(no_repeat(N, T, E, F))
    if no_repeat(N, T, E, F):
        return()

    # how_long()
    P_opt = genetic_opt()
    P_opt = opti_parcours_local(P_opt)
    prediction(P_opt)
    for Fi in range(len(E_liste)):
        F = E_liste[Fi]
        F, E = E, F
        gamma_draw(P_opt)
        F, E = E, F
    """N = 5
    T = 100
    E = "cos(x)"
    F = "cos(x)"
    P = opti_parcours_local()
    gamma_draw(P)
    P = opti_parcours_local()
    gamma_draw(P)
    res = benchmark()
    file = open("results.txt", "w")
    file.write(str(res))
    file.close()"""


i,j,k = sys.argv[1:]
i,j,k = int(i),int(j),int(k)
main(i, j, k)
