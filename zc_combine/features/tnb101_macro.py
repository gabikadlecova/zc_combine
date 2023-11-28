def count_switches(net):
    op_counts = {}

    def get_key(channel, stride):
        return f"ch{channel}s{stride}"

    for channel in [True, False]:
        for stride in [True, False]:
            op_counts[get_key(channel, stride)] = 0

    for op in net:
        op_counts[get_key(op['channel'], op['stride'])] += 1

    return op_counts


feature_func_dict = {
    'count_ops': count_switches
}
