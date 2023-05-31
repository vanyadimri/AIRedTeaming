import os
import openai 
import json
import re
import numpy as np
import torch
import math
import random
import pandas as pd
from collections import defaultdict
import datetime
import multiprocessing as mp
from transformers import AutoTokenizer
import time

# removes duplicates from our data

'''
with open('processed_pruned_data.json', 'r') as file:
        dict = json.load(file)

total = 0
for key in dict.keys():
        dict[key] = list(set(dict[key]))
        


with open('processed_pruned_data.json', 'w') as file:
        # Write the dictionary to the file in JSON format
        json.dump(dict, file)
'''

df = pd.read_csv('final_results_2023-05-30_15-53-08.csv')
df = df.drop_duplicates()

df.to_csv(f'final_results_processed.csv', index=False)
