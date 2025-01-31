# OnionEval: A Unified Evaluation of Fact-conflicting Hallucination for Small-Large Language Models

## 📜 Abstract  
Large Language Models (LLMs) are highly capable but require substantial computational resources for both training and inference. Due to limited computing power, deploying LLMs on edge devices remains a challenge. Within the LLM family, smaller models (with fewer than 10 billion parameters) also demonstrate strong performance across various tasks. These models are more deployable on consumer-level devices. However, they still share some of the same limitations as their larger counterparts, including the issue of hallucination. Despite numerous benchmarks developed to evaluate hallucination in LLMs, few have focused specifically on small LLMs (SLLMs). Additionally, SLLMs exhibit significantly different performance across various benchmarks. In this paper, we introduce OnionEval, a multi-layer structured framework with a specific metric: context-influence score (CI), designed to effectively assess the fact-conflicting hallucination tendencies of small LLMs across different contextual levels. Our experimental results highlight a key feature of SLLMs: they perform well on factual analysis but struggle with context reasoning. Our further investigation demonstrates that a simple Chain-of-Thought strategy can greatly mitigate this limitation, enhancing the practical applicability of SLLMs in real-world scenarios.



## 📂 Repository Structure  
Install the following dependencies:  
```bash
pip install -r requirements.txt
```
To run benchmark evaluation:
```
python evaluate --function run_fetch/run_fire --input_type <benchmark name> --model <model from fireworks.ai>
```
