import collections
import matplotlib.pyplot as plt
import re
import datetime as dt

# plotting distribution of length of strings
def plotDistribution(strings):
    len_of_string = [len(re.split('\s|, |\.|:', string)) for string in strings]
    print(len_of_string[:5])
    print(max(len_of_string))
    print(len(strings))

    dtime = dt.time()
    now = dt.datetime.now()
    now.strftime("%Y-%m-%d %H-%M-%S")

    A = collections.Counter(len_of_string)
    plt.bar(A.keys(), A.values())
    plt.savefig(f'{now}.png')