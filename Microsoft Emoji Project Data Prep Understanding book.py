#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import os 
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, default_data_collator
from transformers import AutoTokenizer, AutoModelForCausalLM

#pip install einops
#package required for downloading transformer 
#https://huggingface.co/datasets/cnn_dailymail link to dataset




# In[2]:


dataset = load_dataset("cnn_dailymail",'3.0.0', split='train')
#Run loading script once located in same directory as python notebook
#Specify configuration version '3.0.0'
#load dataset using script because dataset is very large


# In[3]:


df=pd.DataFrame(dataset)
#cast dataset into pandas dataframe object type 


# In[4]:


df.head(10)


# In[5]:


df.drop(columns='id', inplace=True)
#drop stringid colummn from dataset


# In[6]:


#Download tokenizer associated with "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5",  from_tf=False,trust_remote_code=True)


# In[7]:


tokenizer.pad_token = tokenizer.eos_token


# In[8]:


#Take 1000 examples from our dataframe to use for training
pretask_dataset=df.iloc[0:1000].copy()

#tokenize pretask_dataset 'article' column and turn words into subword tokens for 'article' column stored in new 
#column name 'article_tokenized'

#apply across article column of pretask_dataset tokenizer and store results for each example in corresponing row and new column
#name 'article_tokenized'
pretask_dataset['article_'+'tokenized']=pretask_dataset['article'].apply(lambda x: tokenizer.tokenize(x, truncation=True))

#same procedure except new column name 'highlights_tokenized'
pretask_dataset['highlights_'+'tokenized']=pretask_dataset['highlights'].apply(lambda x: tokenizer.tokenize(x, truncation=True))


#peak at pretask_dataset
pretask_dataset.head(1)

#must use 'lambda x: ' function with tokenizer.tokenize(x) to apply tokenization to all rows 
#no error was outputted
    


# In[9]:


#Create new dataframe with only tokenized data of article and highlights 

columns_to_remove= ['article','highlights']
pretask_dataset_tokenized = pretask_dataset.drop(columns=columns_to_remove, axis=1, inplace=False).astype(str)
pretask_dataset_tokenized


# In[10]:


#Now work on creating pretask labels from keywords to align with "keyword presence" task

df_2=pd.read_csv('keywords.csv')

#make empty list
list1 = []

for word in df_2:
    for j in df_2[word]:
        list1.append(j)

print(list1)
    


# In[11]:


characters_to_remove = '\xa0'

cleaned_list = [''.join(char for char in string if char not in characters_to_remove)for string in list1]

print(cleaned_list)


# In[12]:


#create a list object with keywords that we will count the occurence of as a pretext task goal   

individual_words = []

for text in cleaned_list:
    words= text.split()
    individual_words.extend(words)
    
    
print(individual_words)

       


# In[13]:


#comma taken out of list and each word it's own value 

character_to_remove=','
cleaned_individual_words = [''.join(char for char in string if char not in character_to_remove)for string in individual_words]

print(cleaned_individual_words)


# In[14]:


#pretask goal: Label the instance for each keyword in the list "cleaned_individual_words" amongst 1000 text entry examples from cnn_dailynews

#create empty list to store dataframe
articles_labels_df=pd.DataFrame()

#Loop through list of keywords
#Check if keyword is present amongst all rows for datapoint under article and if so label this as 1 and if not label 0
#create dataframe for each individual keyword and their occurence amongst all samples 
#add individual dataframes to list articles_dataframe 
#concatenate articles_dataframe with articles_labels_df so all individual dataframes are together in one dataframe serving as columns 


for keyword in cleaned_individual_words:
    articles_keyword=pretask_dataset_tokenized['article_tokenized'].str.lower().str.contains(keyword.lower()).astype(int)
    
    articles_keyword=articles_keyword.rename('article_' + keyword)
   
    articles_labels_df[articles_keyword.name] = articles_keyword


# In[15]:


articles_labels_df.head(2)


# In[16]:


#Now we must count occurences of keyword in summary column of our dataframe and then join these two dataframes together
#together these will be the labels of our dataset that is text article entries and their summaries so a total of two features 

#So now use 'highlights' column and count occurences of keywords saved above to make a new column consisting of keyword +'highlights'

highlights_labels_df=pd.DataFrame()

for keyword in cleaned_individual_words:
    highlights_keyword = pretask_dataset_tokenized['highlights_tokenized'].str.lower().str.contains(keyword.lower()).astype(int)
    
    highlights_keyword=highlights_keyword.rename('highlights_' + keyword)
    
    highlights_labels_df[highlights_keyword.name] = highlights_keyword    



# In[17]:


#CHECKPOINT: Use row 198 to check if highlights datapoint for that row contains the word 'hurt'
pretask_dataset.iloc[198, 1]


# In[18]:


#Use mask to retreive rows which have the woord 'hurt' present in their highlights datapoint 

mask=highlights_labels_df['highlights_hurt']==1

examples_with_hurt_highlights=highlights_labels_df[mask]

#We see row 198 does have the keyword 'hurt' present so counting keyword occurences was successful
print("Rows with a value of 1")
examples_with_hurt_highlights


# In[19]:


#Download transformer using "namespace/modelname" and trust_remote_code=True to allow downloading remote software 



#Use class name AutoModelForCausualLM 



phi_model= AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", from_tf=False, trust_remote_code=True)


# In[20]:


pretask_dataset_tokenized.head(1)


# In[21]:


articles_labels_df.columns


# In[22]:


new_df= pd.merge(pretask_dataset_tokenized, articles_labels_df, left_index=True, right_index=True)

new_df_2=pd.merge(new_df, highlights_labels_df, left_index=True, right_index=True)


# In[23]:


new_df_2.head(199)


# In[24]:


from datasets import Dataset


# In[25]:


tokenized_dataset = Dataset.from_pandas(new_df_2)


# In[26]:


features = tokenized_dataset.features
print(features)


# In[27]:


#Making dataset compatible with tranformer library 
#Extracting names of label columns 
contains_labels=[label for label in features if label not in ['article_tokenized', 'highlights_tokenized']]


# In[28]:


print(contains_labels)


# In[29]:


id2label = {idx:contains_labels for idx, contains_labels in enumerate(contains_labels)}
label2id = {contains_labels:idx for idx, contains_labels in enumerate(contains_labels)}


# In[30]:


def preprocess_data(examples):
    #Take text from highlights_tokenized and article_tokenized column to store for example iteration
    highlights_text = examples['highlights_tokenized']
    articles_text = examples['article_tokenized']

    # Encode 'highlights_tokenized'
    encoding_highlights = tokenizer(
        highlights_text,
        padding="max_length",
        truncation=True,
        max_length=128
    )

    # Encode 'article_tokenized'
    encoding_articles = tokenizer(
        articles_text,
        padding="max_length",
        truncation=True,
        max_length=128
    )

    #batch labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in contains_labels}

    #make numpy array of shape required for model
    labels_matrix = np.zeros((len(highlights_text), len(contains_labels)))

    
    for idx, label in enumerate(contains_labels):
        labels_matrix[:, idx] = labels_batch[label]

    #Encoded features
    encoding = {
        'input_ids_highlights': encoding_highlights['input_ids'],
        'attention_mask_highlights': encoding_highlights['attention_mask'],
        'input_ids_articles': encoding_articles['input_ids'],
        'attention_mask_articles': encoding_articles['attention_mask'],
        'labels': labels_matrix.tolist()
    }

    return encoding


# In[31]:


encoded_dataset = tokenized_dataset.map(preprocess_data, batched=True)


# In[32]:


encoded_dataset.set_format('torch')


# In[36]:


train_dataset, test_dataset=encoded_dataset.train_test_split(test_size=0.1)


# In[37]:


#State training arguments
training_args = TrainingArguments(
    output_dir="./",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./",
)


# In[38]:


#Initialize trainer instance with training dataset 
trainer = Trainer(
    model=phi_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


# In[39]:


trainer.train()


# In[ ]:




