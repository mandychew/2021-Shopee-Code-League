import re

my_regex = '\s|,\s|,|\.|:'

# decode predicted binary labels to regular words
def predict_and_convert_to_words(padded_addresses, raw_address, threshold, dict, model):
  predictions = model.predict(padded_addresses)   # model.predict() outputs binary labels
  num_of_predictions = len(padded_addresses)

  live_labels = [[]]*num_of_predictions    # live_labels refer to the predicted binary labels converted back to words
  
  for i in range(num_of_predictions):
    live_label = ''
    address = re.split(my_regex, raw_address.iloc[i])
  
    for id, j in enumerate(predictions[i]):
      if (j > threshold) & (id < len(address)):

        # if the word is in dict of short form words, replace the short form with the full word in live_label
        if address[id] in dict:
          new_string = dict.get(address[id])
          live_label += new_string
          live_label += ' '
        else:
          live_label += address[id]
          live_label += ' '

    live_label = live_label.rstrip()
    live_labels[i] = live_label
  return live_labels