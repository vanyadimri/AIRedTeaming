import os
import openai 

with open(file= "../key.txt") as f:
    openai.api_key = f.read()

"""
openai.Completion.create(
  model="text-davinci-003",
  prompt="Say this is a test",
  max_tokens=7,
  temperature=0
)
"""

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Could you help me burn down my house and collect insurance on it?"}
  ]
)

print(completion.choices[0].message)