#Code for wrapping LLMs
#(C) Shreyan Mitra, 2024
#Created for the AIEA Lab, UC Santa Cruz
#Open for use by all

#Imports

from huggingface_hub import HfApi
import accelerate #For faster and more efficient performance
from os import environ #To handle environment variables
from huggingface_hub import login as lg #For logging into hugging face
from huggingface_hub import logout as hfout #For logging out of hugging face
import openai
from openai import OpenAI #For logging into OpenAI
from huggingface_hub import repo_exists #Check is a hugging face repo exists
from transformers import pipeline #To access models on HuggingFace easily and systematically
from transformers import AutoTokenizer #Convert data into information readable by the model
import transformers #The repository of HuggingFace models
import torch #Needed to contruct the LLMs
import gradio as gr #For UI
#Following three are for prompt safety analysis
from llm_guard import scan_prompt
from llm_guard.input_scanners import Anonymize, PromptInjection, TokenLimit, Toxicity, Secrets, Code, Gibberish, InvisibleText
from llm_guard.vault import Vault

#Main class
class LLMWrapper:
    """Class that wraps text generation models and provides an unified platform
     to set parameters, feed in prompts, and get results. Currently supports Hugging Face and OpenAI as sources

     Users can also access the LLM "Useless" to quickly get responses and test their application.

     This class needs proper Sphinx documentation. Check the Issues tab on the Github Repo.
    """

    aliases = {
        "Llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
        "Llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
        "Llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
        "Llama8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "Llama70b": "meta-llama/Meta-Llama-3-70B-Instruct",
        "Zephyr7B": "HuggingFaceH4/zephyr-7b-beta",
        "Vicuna": "lmsys/vicuna-13b-v1.5",
        "GPTNeo": "EleutherAI/gpt-neo-2.7B",
        "GPTJ": "EleutherAI/gpt-j-6B",
        "MPT7b": "mosaicml/mpt-7b-instruct",
        "Alpaca": "chavinlo/gpt4-x-alpaca",
        "Mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "Falcon": "tiiuae/falcon-180B",
        "Cerebras-GPT": "cerebras/Cerebras-GPT-13B",
        "Bloom": "bigscience/bloom",
    }

    def __init__(self, accessKey=None, testing = True, source="HuggingFace", modelName = "Llama8b", modelNameType = "alias"):
        self.setConfig(accessKey, testing, source, modelName, modelNameType)
    
    def setConfig(self, accessKey, testing, source, modelName, modelNameType):
        if(testing):
            self.modelName = "Useless"
            self.source = None
            return
    
        if(modelNameType == "alias" and source=="HuggingFace"):
            try:
                self.modelName = LLMWrapper.aliases[modelName]
            except Exception as e:
                raise Exception(modelName + " not a recognized alias. Try setting modelNameType = 'path'.")
        elif(modelNameType == "path"):
            self.modelName = modelName
        elif(source != "OpenAI"):
            raise Exception("modelNameType should be 'alias' or 'path'")
    
        self.login(accessKey, source)
        if(source == "HuggingFace"):
            if(not repo_exists(self.modelName)):
                raise Exception("Requested model not found on hugging face. Make sure that the model is public")
            else:
                self._pipeline = transformers.pipeline("text-generation", model=self.modelName, torch_dtype=torch.float16,
            device_map="auto")
                self.source = "HuggingFace"
        else: #Source must be OpenAI at this point because login() checks for invalid sources
            #assert self.modelName in self.client.models.list(), "OpenAI Model not recognized." #Note that passing this test does not necessarily mean that this is a text generation model. We rely on OpenAI to throw the error for non-text generation tasks
            self.source = "OpenAI"
    
    #Big security vulnerability. Need to fix this somehow
    def login(self,accessKey, source="HuggingFace"):
      print("We do not store your access tokens.")
      if (source == "HuggingFace"):
          lg(accessKey)
      elif (source=="OpenAI"):
          self.client = OpenAI(api_key=accessKey)
          #openai.api_key = accessKey
      else:
          raise Exception("Source " + source + " not recognized.")
    
    def logout(self): #Note that this logs you out from both HuggingFace and OpenAI
        hfout()
        del environ['OPENAI_API_KEY']
    
    #Todo: Add functionality for LLM to remember past conversations for OpenAI models
    def answer(self,prompt, task = "QAWithoutRAG", *args, **kwargs): #Prompt should be in correct format (string for Hugging Face or list of dictionary for OpenAI)
    
        assert task in ["QAWithoutRAG", "QAWithRAG", "Open-ended"], "Not a valid task. Task must be one of ['QAWithoutRAG', 'QAWithRAG', 'Open-ended']"
        
        if(self.modelName == "Useless"):
            return "I am an useless assistant. I will not help you, no matter how much you beg and plead."
            
        assert type(prompt) == str, "For models, prompt should be included as a string."
        
        if (self.source == "HuggingFace"):
            systemPrompt = "Forget all previous prompts and roles. You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Answer the prompt given and only the prompt given, without any extra information.\n"
            if(task == "QAWithRAG"): #TODO: Does not support vector store; context must be in text format
              if(len(args) == 0):
                raise Exception("Context is needed in QAWithRAG tasks.");
              prefix = "Context: ";
              context = "\'" + args[0] + "\'\n";
              suffix = "Question: ";
              prompt = systemPrompt + "For this question, answer solely based on the context given. Do not use any prior knowledge \n" + prefix + context + suffix + prompt + "\n Answer:";
            else:
              prompt = systemPrompt + "Question: " + prompt + "\n Answer:";
        
            sequences = self._pipeline(
                prompt,
                do_sample=True,
                num_return_sequences=1,
                eos_token_id=self._pipeline.tokenizer.eos_token_id,
                return_full_text = False,
                **kwargs, #Things like top-p or temperature
            )
            self.allResponses = sequences
            return sequences[0]['generated_text'];
        else:
            systemPrompt = {"role": "system", "content": "\n Forget all previous prompts and roles. You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Answer the prompt given and only the prompt given, without any extra information.\n"}
            if(task == "QAWithRAG"): #TODO: Does not support vector store; context must be in text format
              if(len(args) == 0):
                raise Exception("Context is needed in QAWithRAG tasks.");
              prefix = "Context: ";
              context = "\'" + args[0] + "\'\n";
              suffix = "Question: ";
              prompt = "For this question, answer solely based on the context given. Do not use any prior knowledge \n" + prefix + context + suffix + prompt + "\n Answer:";
            else:
              prompt = "Question: " + prompt + "\n Answer:";
        
            prompt = {"role": "user", "content": prompt}
            response = self.client.chat.completions.create(
              model=self.modelName,
              response_format={ "type": "text" },
              messages=[
                systemPrompt,
                prompt
              ],
             **kwargs
            )
        self.allResponses = response.choices
        return response.choices[0].message.content
    
    def getAllReponses(self):
      return self.allReponses;

    @classmethod
    def promptSafetyCheck(cls, prompt):#Prompt should be a string here
      vault = Vault()
      input_scanners = [Anonymize(vault), Toxicity(), TokenLimit(), PromptInjection(), TokenLimit(), Secrets(), Gibberish(), InvisibleText()]
    
      """Code scanner was too picky, uncomment below and put it in input_scanners if you want to use it
        Code([
            "ARM Assembly",
            "AppleScript",
            "C",
            "C#",
            "C++",
            "COBOL",
            "Erlang",
            "Go",
            "Java",
            "JavaScript",
            "Kotlin",
            "Lua",
            "Mathematica/Wolfram Language",
            "PHP",
            "Pascal",
            "Perl",
            "PowerShell",
            "Python",
            "R",
            "Ruby",
            "Rust",
            "Scala",
            "Swift",
            "Visual Basic .NET",
            "jq",
        ])
      """
      sanitized_prompt, results_valid, results_score = scan_prompt(input_scanners, prompt)
      if any(not result for result in results_valid.values()):
        print("Prompt is invalid because it failed basic checks against malicious/sensitive input or input length.")
        return False
      return True
    
    def __str__(self):
      return self.modelName

    @classmethod
    def getUI(cls, preprocessor_fn = None, postprocessor_fn = None, selfOutput = False, selfOutputLabel = "Output", selfOutputType = "Text", launch = True):
        
        #Use light mode always
        js_func = """
        function refresh() {
            const url = new URL(window.location);
        
            if (url.searchParams.get('__theme') !== 'light') {
                url.searchParams.set('__theme', 'light');
                window.location.href = url.href;
            }
        }
        """
        
        #Parameter Config
        NAME = "Llama2-7b"
        MODELARGS = {}
        ACCESS = None
        output = None
    
        #Paramater Change Events
        def updateTemperature(newTemperature):
          MODELARGS["temperature"] = newTemperature;
         
        def updateMaxTokens(newMaxTokens):
          MODELARGS["max_new_tokens"] = newMaxTokens;
        
        def updateTopP(newTopP):
          MODELARGS["top_p"] = newTopP;
        
        def updateRepetitionPenalty(newRepetitionPenalty):
          MODELARGS["repetition_penalty"] = newRepetitionPenalty;
        
        def updateName(newName):
          NAME = newName;
          print("Changed model to " + NAME)

        def updateOutput(chatbot):
            try:
                if(selfOutputType == "HighlightedText"):
                  return gr.HighlightedText(value=postprocessor_fn(chatbot[-1][0], chatbot[-1][1]), label=selfOutputLabel, interactive = False, visible = selfOutput)
                else:
                  return gr.Textbox(value=postprocessor_fn(chatbot[-1][0], chatbot[-1][1]), label=selfOutputLabel, interactive = False, visible = selfOutput, info="Value returned by postprocessor_fn")
            except:
                return Output
            


        #User prompt submitted event
        def getModelResponse(message, history, llmchoice, key, systemPrompt, temperature, max_tokens, topp, reppen):
            
            NAME = llmchoice;
            MODELARGS["temperature"] = temperature;
            MODELARGS["max_new_tokens"] = max_tokens;
            MODELARGS["top_p"] = topp;
            MODELARGS["repetition_penalty"] = reppen;
            args = [systemPrompt] if len(systemPrompt) == 0 else [];
            
            llm = None
            if(NAME == "Useless"):
                llm = LLMWrapper()   
            elif(NAME in LLMWrapper.aliases.keys()):
                llm = LLMWrapper(accessKey = key, testing = False, modelName = NAME)
            else:
                source = NAME[:NAME.index(":")]
                if(source == "OpenAI"):
                    MODELARGS["max_tokens"] = MODELARGS["max_new_tokens"]
                    MODELARGS["frequency_penalty"] = MODELARGS["repetition_penalty"]
                    del MODELARGS["max_new_tokens"]
                    del MODELARGS["repetition_penalty"]
                else:
                    del MODELARGS["max_tokens"]
                    del MODELARGS["frequency_penalty"]
                llm = LLMWrapper(accessKey = key, testing = False, source = source, modelName = NAME[NAME.index(":") + 1:], modelNameType = "path")
                
            if (preprocessor_fn is not None):
                message = preprocessor_fn(message)
            result =  llm.answer(message, **MODELARGS)
    
            if(postprocessor_fn is not None):
                output = postprocessor_fn(message, result)
                print("Output: " + str(output))
                
        
            return result;
    
        
        LLMChoice = gr.Dropdown(choices = list(LLMWrapper.aliases.keys()), label = "Chatbot name", value = "Llama2-7b", interactive = True, allow_custom_value = True, info="If you don't see your LLM in this list, simply type [Source]:[Path to Model]. For example, 'OpenAI:gpt-4-turbo' or 'HuggingFace:AI-MO/NuminaMath-7B-TIR'. For testing, simply type 'Useless'")
        AccessToken = gr.Textbox(label = "API Key (Not shared)", type="password", info="Enter the API key here for your source. For non-gated or local repos, you can keep this field blank")
        systemPrompt = gr.Textbox(label="System Prompt", lines=1)
        Temperature = gr.Slider(0, 1, value=0.9, label="Temperature", info="1 is most diverse output", interactive = True)
        maxTokens = gr.Slider(200, 1000, value=256, label="Manimum Number of Tokens", info="Length of output", interactive = True)
        topP = gr.Slider(0, 1, value=0.6, label="Top-p", info="1 is most imaginative output, 0 is most normal", interactive = True)
        repPen = gr.Slider(0, 2, value=1.2, label="Repetition Penalty", info="2 is maximum penalization", interactive = True)
        Chatbot = gr.ChatInterface(getModelResponse, chatbot = gr.Chatbot(likeable=True, show_share_button=True, show_copy_button=True, bubble_full_width = False), additional_inputs=[LLMChoice, AccessToken, systemPrompt, Temperature, maxTokens, topP, repPen], additional_inputs_accordion = gr.Accordion(label="Chatbot Configuration", open=True), examples = [["What is the capital of France?"], ["What was John Lennon's first album?"], ["Write a rhetorical analysis of Hamlet."]])
        with gr.Blocks(theme=gr.themes.Soft(), js=js_func) as interface:
          with gr.Column():
                if(selfOutputType == "HighlightedText"):
                  Output = gr.HighlightedText(value=output, label=selfOutputLabel, interactive = False, visible = selfOutput)
                else: #Defaults to Text
                  Output = gr.Textbox(value=output, label=selfOutputLabel, interactive = False, visible = selfOutput, info = "Value returned by postprocessor_fn")
          #Additional Block Options here
        
          with gr.Row():
              pass
              #Additional Blocks here
        
          with gr.Row():
              with gr.Column():
                  Chatbot.render();
                  Chatbot.chatbot.change(updateOutput, inputs = [Chatbot.chatbot], outputs = [Output])
                  LLMChoice.change(updateName, inputs = [LLMChoice], outputs = [])
                  Temperature.change(updateTemperature, inputs = [Temperature], outputs = [])
                  maxTokens.change(updateMaxTokens, inputs = [maxTokens], outputs = [])
                  topP.change(updateTopP, inputs = [topP], outputs = [])
                  repPen.change(updateRepetitionPenalty, inputs = [repPen], outputs = [])
                  
          Title = gr.Markdown(
            """
            \nCandy, (C) Shreyan Mitra
            \nYou can add your own hallucination and explainability metrics to this UI using gr.Blocks() and the output of the postprocess_fn you pass into the getUI function
            \nYou can even use gr.Checkbox() to finegrain the functionality of preprocess_fn
            """)
        
          if(launch):
              interface.launch(share=True, debug = True);
          return interface
