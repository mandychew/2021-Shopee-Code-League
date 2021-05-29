# plotting distribution of length of raw_addresses and of POI
import collections
import matplotlib.pyplot as plt
import re

def plotDistribution(addresses):
    length_of_addresses = [len(re.split('\s|, |\.|:', address)) for address in addresses]
    print(length_of_addresses[:5])
    print(max(length_of_addresses))
    print(len(addresses))

    A = collections.Counter(length_of_addresses)
    plt.bar(A.keys(), A.values())
    plt.show()