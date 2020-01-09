"""
To provide a tool kit of Beam Search
Almost all of methods succeed 'hd' since they require sos_id.
"""

import math
import sys
import numpy as np
import torch
import time
import copy
import utils.tools as tools

class BeamCell():
    def __init__(self, name, hyp, params, cover):
        self.name = name
        self.hyp = hyp
        self.params = params
        self.coverage = cover
        self.prob = 0.0

    def str(self):
        return 'name: ' + str(self.name) + '\nhyp: ' \
                + str(self.hyp) + '\nprob: ' + str(self.prob) + '\ncover: ' + str(self.coverage)

def is_all_end(bl, hd):
    """
    To judge all of BeamCells has <eos>
    BeamCell list -> bool
    """
    eos_id = hd['EOS_ID']
    is_all = True
    for b in bl:
        if not(b.hyp[-1] == eos_id):
            is_all = False
    return is_all


def old_delete_element(dists, name, arg):
    """
    To delete what's once chosen
    """
    for b, dist in dists:
        if b.name == name:
            dist[arg] = 0.0


def delete_element(local_bests, name):
    """
    To delete what's once chosen
    """
    for b, d, a in local_bests:
        if b.name == name and len(d) > 0:
            del d[0]
            del a[0]


def dist_to_local_bests(dist, bw):
    """
    Tensor -> [BeamCell, list(N_best's dist), list(N_best's args)]
    """
    sorted_dist, indices = torch.sort(dist)
    rev_sorted_list = torch.flip(sorted_dist, [0])
    rev_indices = torch.flip(indices, [0])
    max_dists = rev_sorted_list[:bw].tolist()
    max_args = rev_indices[:bw].tolist()
    return [max_dists, max_args]


def calc_coverage(piled_alpha, hd):
    """
    piled_alpha is added alphas so far.
    """
    penalty = 0.0
    #print(piled_alpha)
    penalty = (0.1 > piled_alpha).sum()
    """
    for a in piled_alpha:
        #print(a)
        #if piled_alpha[i] < hd['coverage_tau']:
        if a < 0.05:
            #penalty += math.log(a)
            penalty += 1
    """
    #print(penalty)
    return float(penalty)


def modify_same_element(l):
    """
    I encontered a miracle where two probabilities are completely the same as each other.
    And this causes a really rare error. This finction is to deal with it.
    """
    new_l = []
    for i, e1 in enumerate(l):
        new_e = e1
        for j, e2 in enumerate(l):
            if not(i == j) and e1 == e2:
                l[j] = e1 * 0.999
        new_l.append(new_e)
    return new_l


def get_N_best(dists, bw, hd):
    """
    !!!GREATLY FAST!!!
    beam * dist         * BEAM_WIDTH -> best_beam
    Beam * list(Tensor) * int        -> dict
    """
    local_bests = []
    # local_best: [BeamCell, [list(best_score), list(best_args)]] * BEAM_WIDTH
    if len(dists[0][0].hyp) == 1:
        # At the first time, the obtainable distribution is only one.
        local_bests.append([dists[0][0]] + dist_to_local_bests(dists[0][1], bw))
    else:
        # Else, you have to consider all distributions.
        for b, dist in dists:
            if dist.size(0) > 0:
                local_bests.append([b] + dist_to_local_bests(dist, bw))
            else:
                local_bests.append([b, [], []])
    #print(local_bests)
    local_bests[0][1] = modify_same_element(local_bests[0][1])
    #print(local_bests)
    best_beams = []
    sup_score = 0.0
    for i in range(bw):
        #print(sup_score)
        score = -9999.9
        #print(sup_score)
        best_beam = {'name': -1, 'arg': -1, 'prob': -1}
        #print(local_bests[0][0].str())
        #print(local_bests)
        for b, d, a in local_bests:
            #cover_penalty = 0.001 * calc_coverage(b.coverage[0][0], hd)
            #cover_penalty = 0 * calc_coverage(b.coverage[0][0], hd)
            if len(d) > 0:
                #print(d)
                #this_score = (b.prob + math.log(d[0])) / (len(b.hyp) + 1) + hd['coverage_lambda'] * calc_coverage(b.coverage)
                #print('this_score: ' + str(this_score))
                #print(b.prob)
                #print(math.log(d[0]))
                #print(len(b.hyp))
                #print(calc_coverage(b.coverage[0][0], hd))
                #print(cover_penalty)
                this_score = ((b.prob + math.log(d[0])) / (len(b.hyp) + 1)) #- cover_penalty
                #print('this_score: ' + str(this_score))
                #print()
                #sys.exit()
                #print(this_score)
                #print(sup_score)
                if score < this_score and this_score < sup_score:
                    #print('Hi!!!')
                    score = this_score
                    best_beam['name'] = b.name
                    best_beam['arg'] = a[0]
                    best_beam['prob'] = b.prob + math.log(d[0])
            else:
                #this_score = b.prob / len(b.hyp) + hd['coverage_lambda'] * calc_coverage(b.coverage)
                this_score = (b.prob / len(b.hyp)) #- cover_penalty
                if score < this_score and this_score < sup_score:
                    score = this_score
                    best_beam['name'] = b.name
                    best_beam['arg'] = hd['EOS_ID']
                    best_beam['prob'] = b.prob
            #print(id(normed_score))
            #print(id(score))
            #print(best_beam['arg'])
        best_beams.append(best_beam)
        #print(score)
        #print()
        sup_score = score
        #print(id(sup_score))
        #print(id(score))
        delete_element(local_bests, best_beam['name'])
    #print(best_beams)
    return best_beams

def get_BeamCell(beam_list, name):
    #print(beam_list)
    #print(name)
    needed_beam = None
    for b in beam_list:
        if b.name == name:
            needed_beam = b
            break
    return needed_beam

def make_next_beam_list(beam_list, bests, hd):
    new_beam_list = []
    #print(bests)
    for i, best_dict in enumerate(bests):
        name = best_dict['name']
        arg = best_dict['arg']
        prob = best_dict['prob']
        best_beam = get_BeamCell(beam_list, name)
        new_hyp = best_beam.hyp + [arg]
        new_params = copy.copy(best_beam.params)
        new_cover = copy.copy(best_beam.coverage)
        #new_params = best_beam.params
        #print(best_dict)
        #print(id(best_beam.params['hid']))
        new_beam = BeamCell(i, new_hyp, new_params, new_cover)
        #print(id(new_beam.params['hid']))
        #print(new_beam.str())
        # i: new_name
        #new_beam.hyp.append(arg)
        #print(new_beam.str())
        new_beam.prob = prob
        #print(new_beam.str())
        new_beam_list.append(new_beam)
        #print(best_dict)
        #print(i)
        #print(new_beam.str())
    #print()
    #sys.exit()
    return new_beam_list


def cut_eos(hyp, hd):
    """
    list -> list
    ex. [2, 4, 5, 1, 1, 1, 1] -> [2, 4, 5, 1]
    """
    while(hyp[-1] == hd['EOS_ID']):
        hyp = hyp[:-1]
    return hyp + [hd['EOS_ID']]


def get_hyps(best_beams, n, hd):
    hyps = []
    for i in range(n):
        hyps.append(cut_eos(best_beams[i].hyp, hd))
    return hyps

def print_hyps(hyps, htk, n, hd):
    for i in range(n):
        print(htk + ' ' + ' '.join(tools.int_to_str_list(hyps[i])))


def init_N_beams(n, params, hd):
    beam_list = []
    for i in range(n):
        beam_list.append(BeamCell(i, [hd['SOS_ID']], params))
    return beam_list
