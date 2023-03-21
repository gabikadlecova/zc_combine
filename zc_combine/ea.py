# TODO cite abdelfattah


import random
import copy

import numpy as np

from zc_combine.search_utils import net_from_spec, random_net, NetData, zc_warmup, get_spec_map


# TODO check if it works the same for their spec and my spec (in notebook)
def mutate_spec(old_net: NetData):
    old_spec = old_net.to_spec()

    idx_to_change = random.randrange(len(old_spec))
    entry_to_change = old_spec[idx_to_change]
    possible_entries = [x for x in range(5) if x != entry_to_change]
    new_entry = random.choice(possible_entries)
    new_spec = copy.copy(old_spec)
    new_spec[idx_to_change] = new_entry
    return new_spec


def mutate_spec_zero_cost(old_net: NetData, df, spec_map, zc_scoring):
    possible_specs = []
    old_spec = old_net.to_spec()

    for idx_to_change in range(len(old_spec)):
        entry_to_change = old_spec[idx_to_change]
        possible_entries = [x for x in range(5) if x != entry_to_change]
        for new_entry in possible_entries:
            new_spec = copy.copy(old_spec)
            new_spec[idx_to_change] = new_entry

            new_net = net_from_spec(new_spec, df, spec_map)

            possible_specs.append((zc_scoring(new_net), new_spec))

    if random.random() > 0.75:
        best_new_spec = random.choice(possible_specs)[1]
    else:
        best_new_spec = sorted(possible_specs, key=lambda i: i[0])[-1][1]

    return best_new_spec


def random_combination(iterable, sample_size):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(pool[i] for i in indices)


def _update(net, pool, best_val_accs):
    net_acc = net.get_val_acc()
    pool.append((net_acc, net))

    if net_acc > best_val_accs[-1]:
        best_val_accs.append(net_acc)
    else:
        best_val_accs.append(best_val_accs[-1])


def run_evolution_search(df, zc_warmup_func=None, zc_mutate_func=None, max_trained_models=1000,
                         pool_size=64,
                         tournament_size=10,
                         zero_cost_warmup=0):
    best_valids = [-np.inf]
    pool = []  # (validation, spec) tuples
    num_trained_models = 0
    spec_map = get_spec_map(df)

    # fill the initial pool
    zero_cost_pool = zc_warmup(df, zc_warmup_func, zero_cost_warmup) if zc_warmup_func is not None else None

    for i in range(pool_size):
        if zero_cost_pool is not None:
            net = zero_cost_pool[i][1]
        else:
            net = random_net(df)

        num_trained_models += 1
        _update(net, pool, best_valids)

    # After the pool is seeded, proceed with evolving the population.
    while True:
        sample = random_combination(pool, tournament_size)
        best_net = sorted(sample, key=lambda n: n[0])[-1][1]
        if zc_mutate_func is not None:
            new_net = zc_mutate_func(best_net)
        else:
            new_net = mutate_spec(best_net)

        new_net = net_from_spec(new_net, df, spec_map)
        if new_net is None:
            continue

        num_trained_models += 1
        _update(new_net, pool, best_valids)

        # kill the oldest individual in the population.
        pool.pop(0)

        if num_trained_models >= max_trained_models:
            break

    best_valids.pop(0)
    return best_valids
