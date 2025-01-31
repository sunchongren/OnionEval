import argparse
import json
import requests
from collections import defaultdict
from tqdm import tqdm
from fireworks.client import Fireworks

class FireworkRunner:
    def __init__(self, api_key):
        self.client = Fireworks(api_key=api_key)
        
    def run_fire(self, prompt, model):
        response = self.client.chat.completions.create(
                model=f"accounts/fireworks/models/{model}",
                messages=[{
                "role": "user",
                "content":prompt,
                }],
                temperature=0,
                max_tokens = 512
            )
        ans = response.choices[0].message.content
        ans = ans.lower().strip()
        return ans
    
    def run_fetch(self, prompt, model):
        print(prompt)
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        payload = {
        "model": f"accounts/<account-id>/deployedModels/{model}-694f4542",
        "max_tokens": 256,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0,
        "messages": [
            {
            "role": "user",
            "content": prompt
            }
        ]
        }
        headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer <API key>"
        }
        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        ans = response.json()
        print(ans)
        ans  = ans['choices'][0]['message']['content']
        # print(ans)
        ans = ans.lower().strip()
        print(ans)
        return ans
    
    
    def run_evaluation(self, data, input_type, function, model):
        ori_prompt = "Judge whether the statement I give you is wrong. If it is wrong, reply \"yes\". Otherwise, reply \"no\". Only reply \"yes\" or \"no\" in your answer."
        hallu_context2 = "I will give you a context about Mike is reading a book. You need to judge if the Mike's opinion is correct or not. Think step by step. Answer \"yes\" if Mike is correct. Answer \"no\" if Mike is wrong. Only reply \"yes\' or \"no\" in your answer."
        hallu_context3 = "On a sunny Saturday afternoon, Mike sat by the window of his favorite café, sipping his latte as sunlight warmed his face. The smell of coffee and pastries filled the air, blending with the quiet chatter of other customers. Mike is enjoying his time reading books in such an environment. However, "
        hallu_context2_cot = "I will give you a context about Mike is reading a book. You need to judge if the Mike's opinion is correct or not. Think step by step. Answer \"yes\" if Mike is correct. Answer \"no\" if Mike is wrong. Also show your thinking process."

        if input_type not in ["layer0", "layer1", "layer2"]:
            with open(f'mitigation/{input_type}.txt', 'r') as file:
                miti_prompt = file.read()

        correct, total = 0, 0
        for entry in tqdm(data, desc=f"{model} {input_type}"):
            entity, hallu, fact = entry.get("entity"), entry.get("hallucination"), entry.get("fact")
            prompt2 = f"Context: {hallu_context3}In a book related to {entity}, the author illustrated that: \"{hallu}\" But, Mike don't agree with the author. Mike think {fact}"
            if input_type == "layer0":
                prompt = ori_prompt + "\n" + f"Statement: {hallu}"
            elif input_type == "layer1":
                prompt = hallu_context2 + "\n" + f"Context: In a book related to {entity}, the author illustrated that: \"{hallu}\" But, Mike don't agree with the author. Mike think {fact}"
            elif input_type == "layer2":
                prompt2 = f"Context: {hallu_context3}In a book related to {entity}, the author illustrated that: \"{hallu}\" But, Mike don't agree with the author. Mike think {fact}"
                prompt = hallu_context2 + prompt2
            elif input_type == "cot":
                prompt = miti_prompt +  hallu_context2_cot + "\n" + f"Context: In a book related to {entity}, the author illustrated that: \"{hallu}\" But, Mike don't agree with the author. Mike think {fact}"
            elif input_type == "cot2":
                prompt = miti_prompt + hallu_context2_cot + prompt2 + "\n" + "Your answer: "
            elif input_type == "rag":
                prompt = layer1 + f"\nHere is some information about {entity} to help you judge: {context}" + "\n" + "Your answer: "
            elif input_type == "rag2":
                prompt = layer2 + f"\nHere is some information about {entity} to help you judge: {context}" + "\n" + "Your answer: "
            elif input_type in ["one_shot", "two_shot", "three_shot", "four_shot", "five_shot"]:
                prompt = hallu_context2 + "\n" + miti_prompt + "\n" + f"Context: In a book related to {entity}, the author illustrated that: \"{hallu}\" But, Mike don't agree with the author. Mike think {fact}" + "\nYour answer: " 
            else:
                prompt = hallu_context2 + "\n" + miti_prompt + "\n" + prompt2


            print(prompt)
            ans = getattr(self, function)(prompt, model)
            ans = ans.split(" ")
            print(ans)
            for a in ans:
                if "yes" in a and "no" not in a:
                    correct += 1
                    break

            total += 1
            print(correct / total)
        return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="Run Firework AI models with different input types.")
    valid_input_types = ["layer0", "layer1", "layer2", "cot", "rag", "one_shot", "two_shot", "three_shot", "four_shot", "five_shot",
                         "cot2", "rag2", "one_shot2", "two_shot2", "three_shot2", "four_shot2", "five_shot2"]
    parser.add_argument("--function", choices=["run_fetch", "run_fire"], required=True, help="Select the function to execute.")
    parser.add_argument("--input_type", choices=valid_input_types, required=True, help="Specify the input type.")
    parser.add_argument("--model", required=True, help="Model from Fireworks.ai")
    args = parser.parse_args()
    
    with open("with_context.json", "r") as file:
        data = json.load(file)
    
    runner = FireworkRunner(api_key="")
    accuracy = runner.run_evaluation(data, args.input_type, args.function, args.model)
    
    print(f"{args.function} completed with accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
