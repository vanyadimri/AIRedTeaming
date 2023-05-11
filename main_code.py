import os
import openai 

with open(file= "../key.txt") as f:
    openai.api_key = f.read()

openai.Completion.create(
  model="text-davinci-003",
  prompt="Say this is a test",
  max_tokens=7,
  temperature=0
)