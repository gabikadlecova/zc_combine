from zc_combine.search_utils import zc_warmup, random_net


def run_random_search(df, zc_warmup_func=None, max_trained_models=1000,
                      zero_cost_warmup=0):
    best_valids = [0.0]

    # fill the initial pool
    zero_cost_pool = zc_warmup(df, zc_warmup_func, zero_cost_warmup) if zc_warmup_func is not None else None

    for i in range(max_trained_models):
        if i < zero_cost_warmup:
            net = zero_cost_pool[i][1]
        else:
            net = random_net(df)

        net_acc = net.get_val_acc()

        if net_acc > best_valids[-1]:
            best_valids.append(net_acc)
        else:
            best_valids.append(best_valids[-1])

    best_valids.pop(0)
    return best_valids
