import json
from openai import OpenAI
from tqdm import tqdm

"""
Generate hallucination based on atomic facts
"""

client = OpenAI()


def get_hallu(element):
    prompt = """You need to check if the description given is match the entity in the category. If they don't match, output \"no\". Otherwise, generate a hallucination version of this description.\n 
            For exmaple: Description: Banana is yellow. Entity: blueberry. Category: plant. Your answer should be: no\n
            Another example: Description: Banana is yellow. Entity: Banana. Category: fruit. Your answer should be: Banana is red\n
            No more explainations, only \"no\" or hallucination.
            """
    fact = element["description"]
    category = element["category"]
    entity = element["entity"]
    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt + "\n### " + "Description: " + fact + "Entity: " + entity + "Category: " + category + "###"
            }
        ]
    )
    message = completion.choices[0].message.content
    if message != "no":
        return {"category": category, "entity": entity, "hallucination": message, "fact": fact}
    else:
        return
    
def hallucinate(file_path):
    output = []

    try:
        with open(file_path, 'r')  as file:
            data = json.load(file)
            for index, element in enumerate(tqdm(data, desc="processing")):
                hallu = get_hallu(element)
                if hallu:
                    output.append(hallu)
        
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")

    return output


json_file_path = 'facts.json'
output = hallucinate(json_file_path)

with open('extracted_article_bodies.json', 'w') as file:
    json.dump(output, file, indent=4)

print("Hallucination generated")