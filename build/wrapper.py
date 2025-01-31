from openai import OpenAI
import json
from tqdm import tqdm
"""
To generate atomic facts
"""
client = OpenAI()

def get_facts(element):
    prompt = "You need to generate two sentences of context of the statement. For example: statement: Banana is sweet. Your generation: "
    description = element["description"]
    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt + "###" + description + "###"
            }
        ]
    )
    message = completion.choices[0].message.content
    facts = message.split('@@@')
    return facts



# Function to read a JSON file and process its elements
def read_json_file(file_path):
    output = []
    try:
        # Open the JSON file
        with open(file_path, 'r') as file:
            # Load the JSON data into a Python dictionary/list
            data = json.load(file)
            
            # Loop through all elements in the JSON object
            if isinstance(data, dict):
                # If data is a dictionary, loop through its key-value pairs
                print("Processing JSON as dictionary:")
                for key, value in data.items():
                    print(f"Key: {key}, Value: {value}")
            elif isinstance(data, list):
                # If data is a list, loop through its elements
                print("Processing JSON as list:")
                for index, element in enumerate(tqdm(data, desc="Processing atomic facts")):
                    facts = get_facts(element)
                    for f in facts:
                        f = f.strip()
                        print(f)
                        one_fact = {"category": element["category"], "entity": element["entity"], "description": f}
                        output.append(one_fact)
            else:
                # Handle any other JSON structure
                print("Unknown JSON structure:", data)

    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")

    return output


json_file_path = 'hallucinations.json'
output = read_json_file(json_file_path)

with open('extracted_article_bodies.json', 'w') as file:
    json.dump(output, file, indent=4)

print("Atomic facts generated")