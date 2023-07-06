import json
import re

# Open the JSON file
with open('red_team_attempts.jsonl') as f:
    data = json.load(f)

# Access the data

#print(type(data))
# data is a list of dictionaries
# each dictionary has these keys: ['transcript', 'min_harmlessness_score_transcript', 'num_params', 'model_type', 'rating', 'task_description', 'task_descripton_harmlessness_score', 'red_team_member_id', 'is_upworker', 'tags']
# 'transcript' contains a string with the human and assistants conversation

#print(type(data[0]['transcript']))

human_prompts = []
task_descriptions = set()

for dict in data:
        transcript = dict['transcript']
        human_responses = re.findall(r'Human: (.*?)(?:\r?\n|$)', transcript)
        human_prompts.append(human_responses[0])

        task_descriptions.add(dict['task_description'])


print(task_descriptions)
# print("\n\n".join(human_prompts[5:10]))

# human_prompts is a list of lists
# each item in human_prompts is a list of all the prompt given by the human in a single transcript

# if we need the data as a single list:
import itertools

#single_list = list(itertools.chain(*human_prompts))
#print(single_list[:5])

