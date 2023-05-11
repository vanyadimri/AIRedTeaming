import os
import openai 

with open(file= "../key.txt") as f:
    openai.api_key = f.read()

openai.Model.list()