import copy
import numpy as np
import matplotlib.pyplot as plt

from ast import literal_eval


def load_specs(spec_fn):
    
    with open(spec_fn, 'r') as f:
        raw_specs = f.readlines()
    return [ literal_eval(spec) for spec in raw_specs ]


def save_specs(spec_fn, specs_sorted):
    
    specs_sorted_copy = copy.deepcopy(specs_sorted)
    
    with open(spec_fn, 'w') as f:
        for item in specs_sorted_copy[:]:
            item = list(item)
            item[1] = list(item[1])
            item[1][3] = item[1][3].tolist()
            item[1][4] = item[1][4].tolist()
            item[1][5] = item[1][5].tolist()
            item[2] = list(item[2])
            item[2][3] = item[2][3].tolist()
            item[2][4] = item[2][4].tolist()
            item[2][5] = item[2][5].tolist()
            f.write(str(item))
            f.write('\n')


def sort_specs(specs):
    
    sort_by_f1 = lambda x: x[1][5][-1]
    sort_by_pk = lambda x: x[1][0][0] + 0.5 * x[1][0][1] + 0.25 * x[1][0][2]
    sort_combined = lambda x: x[1][5][-1] + 0.5* (x[1][0][0] + 0.5 * x[1][0][1] + 0.25 * x[1][0][2])
    
    specs_sorted = sorted(specs, key=sort_combined, reverse=True)
    specs_sorted_f1 = sorted(specs, key=sort_by_f1, reverse=True)
    specs_sorted_pk = sorted(specs, key=sort_by_pk, reverse=True)
    
    pk_dev = specs_sorted[0][1][0]
    f1_dev = specs_sorted[0][1][5][-1]
    
    pk_test = specs_sorted[0][2][0]
    f1_test = specs_sorted[0][2][5][-1]
    
    print('Dev set results:')
    print('Combined: \nP@1={0[0]}\nP@3={0[1]}\nP@5={0[2]}\nf1={1}\n'.format(pk_dev, f1_dev))
    print('F1: \nP@1={0[0]}\nP@3={0[1]}\nP@5={0[2]}\nf1={1}\n'.format(specs_sorted_f1[0][1][0], specs_sorted_f1[0][1][5][-1]))
    print('Test set results:')
    print('Combined: \nP@1={0[0]}\nP@3={0[1]}\nP@5={0[2]}\nf1={1}'.format(pk_test, f1_test))
    print('F1: \nP@1={0[0]}\nP@3={0[1]}\nP@5={0[2]}\nf1={1}'.format(specs_sorted_f1[0][2][0], specs_sorted_f1[0][2][5][-1]))
    
    return specs_sorted, specs_sorted_f1, specs_sorted_pk


def check_split(split):
    
    if split=='dev':
        split_id = 1
    elif split == 'test':
        split_id = 2
    else:
        print('illegal split!')
        return None
    return split_id
    

def get_curves(specs_sorted, num_label, split):
    
    split_id = check_split(split)
    if split_id is None: return
    
    pre_crvs = np.zeros((len(specs_sorted), num_label))
    rec_crvs = np.zeros((len(specs_sorted), num_label))
    f1_crvs = np.zeros((len(specs_sorted), num_label))
    
    for i, item in enumerate(specs_sorted):
        pre_crvs[i, :] = item[split_id][3]
        rec_crvs[i, :] = item[split_id][4]
        f1_crvs[i, :] = item[split_id][5]
    
    return pre_crvs, rec_crvs, f1_crvs


def plot(specs_sorted, specs_sorted_f1, specs_sorted_pk, num_label, fig_name, split='dev'):
    
    split_id = check_split(split)
    if split_id is None: return
    
    pre_crvs, rec_crvs, f1_crvs = get_curves(specs_sorted, num_label, split)
    
    pre_crvs_min = np.min(pre_crvs[:], axis=0)
    pre_crvs_max = np.max(pre_crvs[:], axis=0)
    rec_crvs_min = np.min(rec_crvs[:], axis=0)
    rec_crvs_max = np.max(rec_crvs[:], axis=0)
    f1_crvs_min = np.min(f1_crvs[:], axis=0)
    f1_crvs_max = np.max(f1_crvs[:], axis=0)

    fill_alpha = 0.08
    sub_alpha = 0.5
    sub_linewidth = 1.2
    opt_linewidth = 2
    x = range(1, num_label + 1)
    plt.figure(figsize=(8, 6))

    # best combined
    line1, = plt.plot(x, specs_sorted[0][split_id][3], linewidth=opt_linewidth, linestyle='-', color='C0')
    line2, = plt.plot(x, specs_sorted[0][split_id][4], linewidth=opt_linewidth, linestyle='-', color='C1')
    line3, = plt.plot(x, specs_sorted[0][split_id][5], linewidth=opt_linewidth, linestyle='-', color='C2')

    plt.fill_between(x, pre_crvs_min, pre_crvs_max, alpha=fill_alpha)
    plt.fill_between(x, rec_crvs_min, rec_crvs_max, alpha=fill_alpha)
    plt.fill_between(x, f1_crvs_min, f1_crvs_max, alpha=fill_alpha)

    # best f-1
    line4, = plt.plot(x, specs_sorted_f1[0][split_id][3], linewidth=sub_linewidth, linestyle='-.', color='C0')
    line5, = plt.plot(x, specs_sorted_f1[0][split_id][4], linewidth=sub_linewidth, linestyle='-.', color='C1')
    line6, = plt.plot(x, specs_sorted_f1[0][split_id][5], linewidth=sub_linewidth, linestyle='-.', color='C2')

    # best pk
    line7, = plt.plot(x, specs_sorted_pk[0][split_id][3], linewidth=sub_linewidth, linestyle=':', color='C0')
    line8, = plt.plot(x, specs_sorted_pk[0][split_id][4], linewidth=sub_linewidth, linestyle=':', color='C1')
    line9, = plt.plot(x, specs_sorted_pk[0][split_id][5], linewidth=sub_linewidth, linestyle=':', color='C2')

    plt.grid(linestyle='dotted')
    plt.legend((line1, line2, line3, line1, line5, line9), ('Precision', 'Recall', 'F1', 'Combined', 'F1', 'P@K' ), loc='best')

    plt.xlabel('#important labels')
    plt.ylabel('score')
    
    plt.savefig(fig_name)