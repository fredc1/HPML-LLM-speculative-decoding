# HPML-LLM-speculative-decoding
Experimenting with speculative decoding for LLM inference

Startup Instructions
* Find a instance to run experiments on (e.g. GCP instance). Must have a GPU
* (Optional) upload your laptop public key to the ~/.ssh/authorized_keys file, ssh into your instance with VScode remote explorer for nice editor experience
* Add the public key of your instance to github and clone the repo to your instance:
'''bash
git clone git@github.com:fredc1/HPML-LLM-speculative-decoding.git
'''
* Install dependencies:
'''bash
pip install sentencepiece huggingface_hub
'''
* Go to (https://huggingface.co/meta-llama/Llama-2-7b) and follow instructions to get license from Meta and subsequently from the huggingace repo itself
* Upload your instance SSH pub key to huggingface.co so the Meta authorization will be reckognized when you pull the weights
* Download llama weights (in the parent dir of this code repo):
'''bash
hugginface-cli login
'''
'''bash
git clone git@hf.co:meta-llama/Llama-2-7b-chat-hf
'''
'''bash
git clone git@hf.co:meta-llama/Llama-2-13b-chat-hf
'''
'''bash
git clone git@hf.co:meta-llama/Llama-2-70b-chat-hf
'''

