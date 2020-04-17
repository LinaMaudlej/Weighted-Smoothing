import matplotlib.pyplot as plt
import numpy as np
import re


get_text_res_re = re.compile('.+Top1: \d+/\d+ \(([\w.]+)%\), Top5: \d+/\d+ \(([\w.]+)%\)')
def clean_test_res(txt):
    m = get_text_res_re.match(txt)
    return float(m.group(1)), float(m.group(2))

files = []

noise_file_names = [1, 2, 3, 4, 5]
noise_values = [(n, n*0.1) for n in noise_file_names]
vote_values = [('05', 0.05), ('10', 0.1), ('15', 0.15), ('20', 0.2), ('25', 0.25)]
vote_values = [('10', 0.10), ('20', 0.20)]

for n_txt, n in noise_values:
    files.append((f'output\\multiple\\out_boris{n_txt}_e_base_attack.txt', n, 'Baseline'))
    for t_txt, t in vote_values:
        print(n, t)
        files.append((f'output\\out_boris{n_txt}_e_cpt{t_txt}_attack.txt', n, t))

tests = {}

for filename, noise, thresh in files:
    with open(filename, 'r') as f:
        key_test = f'{thresh}, {noise}, Test'
        key_adv = f'{thresh}, {noise}, Adversarial'
        if key_test not in tests:
            tests[key_test] = []
        if key_adv not in tests:
            tests[key_adv] = []
        for l in f.readlines():
            if l.startswith('Test set'):
                top1, top5 = clean_test_res(l)
                tests[key_test].append((top1, top5))
            elif l.startswith('Adverserial set'):
                top1, top5 = clean_test_res(l)
                tests[key_adv].append((top1, top5))

for t, res in sorted(tests.items(), key = lambda d: d[0]):
    top1s = np.array([x for x, _ in res] , dtype=float)
    top5s = np.array([x for _, x in res] , dtype=float)
    print(f'{t}, {np.mean(top1s):.2f}+-{np.std(top1s):.2f}, {np.mean(top5s):.2f}+-{np.std(top5s):.2f}')

