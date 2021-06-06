# Address Elements Extraction
An AI model to extract key address elements from unformatted Indonesian addresses.

## Table of Contents
- [General Info](#general-info)
- [Technologies](#technologies)
- [Features](#features)
- [Project Status](#project-status)
- [Room for Improvement](#room-for-improvement)
- [Acknowledgements](#acknowledgements)

## General Info
This project was created for the competition: 2021 Shopee Code League (Data Science category). 

The project's purpose was to **develop a model to precisely extract the key address elements from the addresses** that Shopee receives. These key address elements are **Point of Interest (POI) Names and Street (ST) Names**. 
However, the addresses Shopee receives are unstructured and in free text format, not following a certain pattern. Thus, it is important that the model can precisely extract key address elements. The key address elements can be geocoded to obtain geographic coordinates to ship Shopee products to their customers - quickly and accurately.

For more information, please refer to this [Kaggle page](https://www.kaggle.com/c/scl-2021-ds).

## Technologies
Project is created with:
- Python version 3.8.5
- Tensorflow version 2.4.1

## Features
- Creating binary labels of addresses
- Created dictionary of short form and full words in addresses
	- prediction will have the full word if the raw address uses short form words
- Tokenization and Padding of addresses
- Creating dictionary of short-form and long-form words
- Keras Sequential model with the following layers:
	- Embedding
	- 2 Bidirectional LSTM layers
	- Flatten
	- 2 Dense layers

## Project Status
Project is _in progress_.

## Room for Improvement
Room for improvement:
- adding Subword Tokenization
- calling a 3rd-party API to check that output is correct
- output doesn't include correct punctuation (e.g. comma, hyphen, period)

To do:
- Automating Hyperparameter Tuning
	- using Bayesian Hyperparameter with Hyperopt
	- or Genetic Algorithm with TPOT

## Acknowledgements
- This project was inspired by the [2021 Shopee Code League](https://careers.shopee.sg/codeleague/) (Data Science category)
- Many thanks to my teammate in the competition
