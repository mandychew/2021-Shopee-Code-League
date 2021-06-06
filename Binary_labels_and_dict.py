import re
import numpy as np

max_len_address = 25   #len(max(raw_df_withPOI['raw_address']))

# create binary labels for addresses; create dictionary of short form words used in addresses
def binary_labels_and_dict(addresses, desired_output, dict, my_regex='\s|,\s|,|\.|:'):
    num_of_binary_labels = len(addresses)
    labels_binary = [[]]*num_of_binary_labels

    # for each word in 'raw_address', match it to words in 'POI/street' column.
    # if there's a match, add '1' to labels_binary; if no match, add '0' 
    for i in range(num_of_binary_labels):
        label_binary = [0 for _ in range(max_len_address)]
        words = re.split(my_regex, desired_output[i])
        address = re.split(my_regex, addresses[i])

        for word in words:
            for j, ad in enumerate(address):
                if word.startswith(ad):
                    # compare each word in address with the desired_output. 
                    # if the word in desired_output starts with the word in address, means that the word in address is in short form.
                    # add the short form and full word in dict ('short form': 'full word')
                    if (ad != word) & ((word in dict.values())==False):
                        dict[ad] = word
                    if j<max_len_address:
                        label_binary[j] = 1
                        break

        labels_binary[i] = label_binary
    
    labels_binary = np.array(labels_binary)
    return labels_binary, dict
    print(labels_binary[:5])
    print(dict(list(dict.items())[0:5]))
    print(len(dict))

# filter out words that contain only digits or have 0 characters from dict
def filter_dict(mydict):
    filter_dict_num = {k: mydict[k] for k in sorted(mydict.keys()) if not any(map(str.isdigit, k))}
    filter_dict_len = {k: filter_dict_num[k] for k in sorted(filter_dict_num.keys()) if len(k)>0}
    return filter_dict_len