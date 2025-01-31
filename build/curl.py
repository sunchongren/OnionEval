import json
import subprocess

"""
To curl knowledge graphbased on entities
"""

# Base URL and API key setup for cURL
base_url = "https://kgsearch.googleapis.com/v1/entities:search?query="
api_key = "&key=<API key>"
headers = "--header 'Accept: application/json' --compressed"

# Function to execute a cURL command and parse the JSON response
def run_curl_and_extract(entity):
    curl_command = f"curl '{base_url}{entity}{api_key}' {headers}"
    try:
        result = subprocess.run(curl_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        article_body = data["itemListElement"][0]["result"]["detailedDescription"]["articleBody"]
        return article_body
    except (json.JSONDecodeError, KeyError, IndexError):
        print(f"Failed to extract articleBody for entity: {entity}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"subprocess error: {e.stderr}")
        return None

# Load the JSON data from the file
with open('./entities.json', 'r') as file:
    data = json.load(file)

outputs = []  # List to store the outputs
id_counter = 0  # Counter for ID assignment

# Iterate over each category and entity within
for category, entities in data.items():
    for entity in entities:
        article_body = run_curl_and_extract(entity)
        if article_body:
            print(entity, "*", article_body)
            outputs.append({"id": id_counter, "entity": entity, "description": article_body, "category": category})
            id_counter += 1

# Save the extracted data to a JSON file
with open('extracted_article_bodies.json', 'w') as file:
    json.dump(outputs, file, indent=4)

print("Article bodies extracted and saved successfully.")
