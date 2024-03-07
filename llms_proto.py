#%%
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, LlamaTokenizer
import os
from huggingface_hub import login
from dotenv import load_dotenv, find_dotenv
# Load environment variables
_ = load_dotenv(find_dotenv())  
# Adjust environment variable names as needed
API_TOKEN = os.environ['API_TOKEN']
login(token = API_TOKEN)

#%%
config = PeftConfig.from_pretrained("BrainGPT/BrainGPT-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = PeftModel.from_pretrained(model, "BrainGPT/BrainGPT-7B-v0.1")


# %% Simple example
# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# Example prompt
prompt = "What does broca's area do"
# Tokenize the input 
inputs = tokenizer(prompt, return_tensors="pt")
# Generate a response from the model
output_sequences = model.generate(input_ids=inputs['input_ids'],
                                  max_length=50,  # Adjust max length as needed 
                                  num_return_sequences=1) 
# Decode the generated response
response = tokenizer.decode(output_sequences[0])
# Print the response
print(response)

# %% Assessing the empirical nature of a paper
# Load the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

title = 'Title:' + 'Discriminating nonfluent/agrammatic and logopenic PPA variants with automatically extracted morphosyntactic measures from connected speech'
abstract = 'Abstract:' + 'Morphosyntactic assessments are important for characterizing individuals with nonfluent/agrammatic variant primary progressive aphasia (nfvPPA). Yet, standard tests are subject to examiner bias and often fail to differentiate between nfvPPA and logopenic variant PPA (lvPPA). Moreover, relevant neural signatures remain underexplored. Here, we leverage natural language processing tools to automatically capture morphosyntactic disturbances and their neuroanatomical correlates in 35 individuals with nfvPPA relative to 10 healthy controls (HC) and 26 individuals with lvPPA. Participants described a picture, and ensuing transcripts were analyzed via part-of-speech tagging to extract sentence-related features (e.g., subordinating and coordinating conjunctions), verbal-related features (e.g., tense markers), and nominal-related features (e.g., subjective and possessive pronouns). Gradient boosting machines were used to classify between groups using all features. We identified the most discriminant morphosyntactic marker via a feature importance algorithm and examined its neural correlates via voxel-based morphometry. Individuals with nfvPPA produced fewer morphosyntactic elements than the other two groups. Such features robustly discriminated them from both individuals with lvPPA and HCs with an AUC of .95 and .82, respectively. The most discriminatory feature corresponded to subordinating conjunctions was correlated with cortical atrophy within the left posterior inferior frontal gyrus across groups (pFWE < .05). Automated morphosyntactic analysis can efficiently differentiate nfvPPA from lvPPA. Also, the most sensitive morphosyntactic markers correlate with a core atrophy region of nfvPPA. Our approach, thus, can contribute to a key challenge in PPA diagnosis.'
paper = title + abstract
prompt = f"""Is this paper {abstract} empirical neuropsychology?"""
# Tokenize the input 
inputs = tokenizer(prompt, return_tensors="pt")

# Generate a response from the model
output_sequences = model.generate(input_ids=inputs['input_ids'],
                                  max_length=500,  # Adjust max length as needed 
                                  num_return_sequences=1) 

# Decode the generated response
response = tokenizer.decode(output_sequences[0])
# Print the response
print(response)


#%% Potential prompt improvements
prompt = f"""
Based on the title ("{title}") and abstract ("{abstract}") provided above, please evaluate if this is an empirical neuropsychology paper. 
1. Empirical Evidence: Does the paper describe empirical research? Empirical research is characterized by data collection from observations or experiments. Look for mentions of methodologies like experiments, surveys, case studies, observations, or clinical trials.
2. Research Focus: Is the focus of the paper on neuropsychology? Neuropsychology involves the study of the relationship between brain function and behavior. Indicators include the study of cognitive processes, brain injury, neurological conditions, or psychological effects of brain disorders.
3. Inclusion of Patients with Neurological Conditions: Does the paper involve patients with neurological conditions? This can be identified by the mention of specific conditions (e.g., Alzheimer's, Parkinson's, epilepsy), brain injuries, or any other neurological impairments, along with the study or treatment of these patients.
4. Analysis and Conclusion: Does the analysis and conclusion section provide insights derived from the study of patients with neurological conditions? This involves looking for results or discussions that specifically address findings related to neurological conditions.
"""