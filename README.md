# CandyLLM 🍬

A simple, easy-to-use framework for HuggingFace and OpenAI text-generation models. The goal is to eventually integrate other sources such as custom large language models (LLMs) as well to create a coherent UI.

This is a work-in-progress, so pull-requests and issues are welcome! We try to keep it as stable as possible though, so people installing this library do not have any problems. 

If you use this library, please cite Shreyan Mitra.

With all the administrivia out of the way, here are some examples of how to use the library. We are still setting up the official documentation. The following examples show some use cases, or tasks, and how an user of llm-wrapper would invoke the model of their choice.

## Install package
```
pip install CandyLLM
```

## Task: Fetch Llama3-8b and run it with default parameters on a simple QA Prompt without retrieval augmented generation

```python
from CandyLLM import*
myLLM = LLMWrapper("MY_HF_TOKEN", testing=False)
myLLM.answer("What is the capital of Uzbekistan?") #Returns Tashkent
```
This behavior is due to the fact that the default model is Llama3-8b

## Task: Fetch Llama2-7b and run it with tempereature = 0.6 on an QA Prompt with retrieval augmented generation
```python
from CandyLLM import*
myLLM = LLMWrapper("MY_HF_TOKEN", testing=False, modelName = "Llama2-7b") #or myLLM = LLMWrapper("MY_HF_TOKEN", testing=False, modelName = "meta-llama/Llama-2-7b-chat-hf", modelNameType="path")
myLLM.answer("What is the capital of Funlandia?", task="QAWithRAG", "The capital of Funlandia is Funtown", temperature=0.6) #Returns Funtown
```

## Task: Fetch GPT-4 and run it with presence_penalty = 0.5 on an Open-Ended Prompt
```python
from CandyLLM import*
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
LLMWrapper.promptSafetyCheck("Is 1010 John Doe's social security number?") #Returns false to indicate unsafe prompt
```

## Change Config
Want to use a different model? No need to create another wrapper.
```python
myLLM = LLMWrapper(...) #Create some LLM wrapper
myLLM.setConfig("MY_TOKEN", testing = False, source="HuggingFace", modelName = "Mistral", modelNameType = "alias") #Tada: a changed LLM wrapper
```

## Dummy LLM
Sometimes, you don't want to spend the time and money to make api calls to an actual LLM, especially if you are testing an UI or an integration of a chat service. Dummy LLMs to the rescue! Our dummy LLM is called "Useless" and it will return answers immediately with very little computation spent (granted, the results it gives are useless - but, hey, what did you expect? 😃)

## CandyUI
CandyUI is the user interface of CandyLLM. It provides a chatbot, a dropdown for choosing the LLM to use, parameter configs for the LLM, and the option to apply post-hoc and pre-hoc methods to the user prompt and LLM output. CandyUI can be integrated into and communicate with a larger UI with custom functions, or you can use the ``selfOutput`` option for the custom post-hoc metrics to be displayed within CandyUI itself.

For example, running
```python
def postprocess(message, response):
    #Sample postprocessor_fn which just returns the difference in length between LLM response and user prompt
    return len(response) - len(message)
x = LLMWrapper.getUI(postprocessor_fn = postprocess, selfOutput = True, selfOutputLabel = "Length Difference")
```
deploys the following webpage:

![Screen Shot 2024-07-17 at 11 53 53 AM](https://github.com/user-attachments/assets/3e0bee23-4cad-427d-8c74-68057c033844)



You can also change how the output is shown. For example, for explainability purposes, you might want to set ```selfOutputType = "HighlightedText"```:

```python
def postprocess(message, response):
    #Randomly assigns importance scores to words in the user prompt
    importantWords = []
    for word in message.split():
        importantWords.append((word, "important")) if len(word) > 3 else importantWords.append((word, "unimportant"))
    return importantWords
x = LLMWrapper.getUI(postprocessor_fn = postprocess, selfOutput = True, selfOutputLabel = "Important Words", selfOutputType = "HighlightedText")

```
The UI now looks like this:
![Screen Shot 2024-07-17 at 11 50 43 AM](https://github.com/user-attachments/assets/83a3f3ae-a566-4fa1-aa9e-a9b3f8751e80)


