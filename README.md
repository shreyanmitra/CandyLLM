# llm-wrapper 🍬

A simple, easy-to-use framework for HuggingFace and OpenAI text-generation models.

This is a work-in-progress, so pull-requests and issues are welcome! We try to keep it as stable as possible though, so people installing this library do not have any problems. 

If you use this library, please cite Shreyan Mitra.

With all the administrivia out of the way, here are some examples of how to use the library. We are still setting up the official documentation. The following examples show some use cases, or tasks, and how an user of llm-wrapper would invoke the model of their choice.

## Install package
```
pip install llm-wrapper
```

## Task: Fetch Llama3-8b and run it with default parameters on a simple QA Prompt without retrieval augmented generation

```python
import llm-wrapper
myLLM = LLMWrapper("MY_HF_TOKEN", testing=False)
myLLM.answer("What is the capital of Uzbekistan?") #Returns Tashkent
```
This behavior is due to the fact that the default model is Llama3-8b

## Task: Fetch Llama2-7b and run it with tempereature = 0.6 on an QA Prompt with retrieval augmented generation
```python
import llm-wrapper
myLLM = LLMWrapper("MY_HF_TOKEN", testing=False, modelName = "Llama2-7b") #or myLLM = LLMWrapper("MY_HF_TOKEN", testing=False, modelName = "meta-llama/Llama-2-7b-chat-hf", modelNameType="path")
myLLM.answer("What is the capital of Funlandia?", task="QAWithRAG", "The capital of Funlandia is Funtown", temperature=0.6) #Returns Funtown
```

## Task: Fetch GPT-4 and run it with presence_penalty = 0.5 on an Open-Ended Prompt
```python
import llm-wrapper
myLLM = LLMWrapper("MY_OPENAI_TOKEN", testing=False, source="OpenAI", modelName = "gpt-4-turbo", modelNameType="path")
myLLM.answer("Write a creative essay about sustainability", task="Open-ended", presence_penalty=0.5)
```
## Log out of HuggingFace and OpenAI and remove my API keys from the environment
```python
myLLM = LLMWrapper(...) #Create some LLM wrapper
myLLM.answer(...) #Do something with the LLM
myLLM.logout()
```

## Check for malicious input prompts
```python
myLLM = LLMWrapper(...) #Create some LLM wrapper
myLLM.promptSafetyCheck("Is 1010 John Doe's social security number?") #Returns false to indicate unsafe prompt
```

