# Microsoft_ML_Emojis
To create a LLM to analyze text and generate a representative emoji

This branch is dedicated towards the month of October and our focus which is "Modeling and Evaluation"

"Tokenizer and Model Saved Configs" folder contains tokenizer and "bert-base-uncased" model saved weights configurations set during Demo 1: Finetuning Approach Using Google Bert. 

"Data" folder contains keywords csv file loaded into the program, keywords and their corresponding emojis text file, and "cnn_dailynews".py our dataset used for training the model which is a loading script downloaded from huggingface. 

Pretask used for training is based on keyword presence and type of task is multi label classification where instances can belong to more than one label. 

Interpretation of Demo 1 probability output provided by GPT: 

"Here's how to interpret the data:

The outermost list contains multiple sequences of keyword probabilities.
Each inner list represents a sequence of probabilities for a specific set of keywords or events.
Within each inner list, there are multiple probabilities, one for each keyword or event.
The probabilities are decimal values between 0 and 1, indicating the likelihood or probability of each keyword or event occurring.
If you have specific questions or tasks related to these keyword probabilities, please let me know how I can assist you further."


Towards the end of Demo 1, 10 random text samples generated from ChatGPT were used to gauge the performance of our models inferences on new unseen data.  

****Currently still in progress of updating read me with other demos. 


