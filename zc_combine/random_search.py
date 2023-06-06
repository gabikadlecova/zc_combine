# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


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
