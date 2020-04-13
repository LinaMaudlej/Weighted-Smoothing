import matplotlib.pyplot as plt
import re
from collections import namedtuple

ResultItem = namedtuple('TestRes', ['m', 'NoiseMagnitude', 'TestRes', 'OtherFalgs'])

get_text_res_re = re.compile('.+Top1: \d+/\d+ \(([\w.]+)%\), Top5: \d+/\d+ \(([\w.]+)%\)')
def clean_test_res(txt):
    m = get_text_res_re.match(txt)
    return float(m.group(1)), float(m.group(2))

tests = []
tests.append(ResultItem(m=512, NoiseMagnitude=0.2, TestRes='Test set: Average loss: 3.7071, Top1: 1056/10000 (10.56%), Top5: 5132/10000 (51.32%)', OtherFalgs='Weight Noise, CPNI, MCPredict, EPGD'))
tests.append(ResultItem(m=512, NoiseMagnitude=0.4, TestRes='Test set: Average loss: 3.7071, Top1: 1056/10000 (11.56%), Top5: 5132/10000 (53.32%)', OtherFalgs='Weight Noise, CPNI, MCPredict, EPGD'))
tests.append(ResultItem(m=512, NoiseMagnitude=0.8, TestRes='Test set: Average loss: 3.7071, Top1: 1056/10000 (12.56%), Top5: 5132/10000 (54.32%)', OtherFalgs='Weight Noise, CPNI, MCPredict, EPGD'))
tests.append(ResultItem(m=512, NoiseMagnitude=0.6, TestRes='Test set: Average loss: 3.7071, Top1: 1056/10000 (13.56%), Top5: 5132/10000 (55.32%)', OtherFalgs='Weight Noise, CPNI, MCPredict, EPGD'))
tests.append(ResultItem(m=256, NoiseMagnitude=0.2, TestRes='Test set: Average loss: 3.7071, Top1: 1056/10000 (12.56%), Top5: 5132/10000 (56.32%)', OtherFalgs='Weight Noise, CPNI, MCPredict, EPGD'))
tests.append(ResultItem(m=256, NoiseMagnitude=0.4, TestRes='Test set: Average loss: 3.7071, Top1: 1056/10000 (13.56%), Top5: 5132/10000 (57.32%)', OtherFalgs='Weight Noise, CPNI, MCPredict, EPGD'))
tests.append(ResultItem(m=256, NoiseMagnitude=0.8, TestRes='Test set: Average loss: 3.7071, Top1: 1056/10000 (14.56%), Top5: 5132/10000 (58.32%)', OtherFalgs='Weight Noise, CPNI, MCPredict, EPGD'))
tests.append(ResultItem(m=256, NoiseMagnitude=0.6, TestRes='Test set: Average loss: 3.7071, Top1: 1056/10000 (15.56%), Top5: 5132/10000 (59.32%)', OtherFalgs='Weight Noise, CPNI, MCPredict, EPGD'))


series = {}
for t in tests:
    key = f'{t.OtherFalgs}, m={t.m}'
    top1 , top5 = clean_test_res(t.TestRes)
    if key not in series:
        series[key] = [(t.NoiseMagnitude, top1, top5)]
    else:
        series[key].append((t.NoiseMagnitude, top1, top5))
        series[key].sort(key=lambda d: d[0])

fig, (ax1, ax2) = plt.subplots(1,2)
plt.ion()
for t, v in series.items():
    xs = [x for x,_,_ in v]
    top1s = [y for _,y,_ in v]
    top5s = [y for _,_,y in v]
    ax1.plot(xs, top1s, label=f'Top1 {t}')
    ax2.plot(xs, top5s, label=f'Top5 {t}')

ax1.set(ylim=(0, 100))
ax2.set(ylim=(0, 100))
ax1.xlabel('Noise magnotude')
ax1.ylabel('Accuracy %')
ax2.xlabel('Noise magnotude')
ax2.ylabel('Accuracy %')
fig.legend()
plt.draw()
