from setuptools import setup, find_namespace_packages

setup(name='llm-wrapper',
      version='2.9.2', 
      description='llm-wrapper: Unified framework for HuggingFace and OpenAI Text-generation Models',
      long_description=open("README.md", "r", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      keywords="llms transformers mistral llama falcon gpt-4 alpaca",
      url="https://github.com/shreyanmitra/llm-wrapper",
      author = "Shreyan Mitra",
      install_requires=[
        "huggingface_hub",
        "accelerate",
        "os",
        "openai",
        "transformers",
        "torch",
        "llm_guard"
      ],
      include_package_data=True,
      package_data={'': ['static/*']},
      packages=["llm-wrapper"],
      )
