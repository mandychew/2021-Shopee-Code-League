import re

my_regex = '\s|,\s|,|\.|:'

def predict_and_convert_to_words(live_to_predict, raw_address, threshold, dictionary, model):
  live_prediction = model.predict(live_to_predict)
  number_of_live_labels = len(live_to_predict)
  # print(number_of_live_labels)

  live_labels = [[]]*number_of_live_labels
  
  # decode live_labels to regular words
  for i in range(number_of_live_labels):
    live_label = ''
    address = re.split(my_regex, raw_address.iloc[i])
  
    for id, j in enumerate(live_prediction[i]):
      if (j > threshold) & (id < len(address)):
        if address[id] in dictionary:
          # print(id, j, address[id])
          new_string = dictionary.get(address[id])
          live_label += new_string
          live_label += ' '
        else:
          # print(id, j, address[id])
          live_label += address[id]
          live_label += ' '
    # print(live_label_POI)
    live_label = live_label.rstrip()
    live_labels[i] = live_label
  return live_labels