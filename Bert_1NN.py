import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

# Function to get sense definition
def get_sense_definition(sense_key):
    synset = wn.lemma_from_key(sense_key).synset()
    return synset.definition()

import xml.etree.ElementTree as ET

def load_semcor_data(xml_file, key_file):
    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    sense_keys = {}
    with open(key_file, 'r') as f:
        for line in f:
            fields = line.strip().split()
            instance_id = fields[0]
            sense_keys[instance_id] = fields[1:]

    sentences,pos,word_senses=[],[],[]
    for sentence in root.findall('.//sentence'):
        sentence_words,sentence_pos,sentence_ws = [],[],[]
        for word in sentence.findall('.//instance'):
            word_text = word.attrib['lemma']
            word_pos = word.attrib['pos']
            word_id = word.attrib['id']
            word_sense = sense_keys.get(word_id, None)
            sentence_words.append(word_text)
            sentence_pos.append(word_pos)
            sentence_ws.append(word_sense[0])
        sentences.append(sentence_words)
        pos.append(sentence_pos)
        word_senses.append(sentence_ws)

    return sentences,pos,word_senses

# Load SemCor data
X,Y,Z = load_semcor_data('/kaggle/input/wsd-training/semcor.data.xml', '/kaggle/input/wsd-training/semcor.gold.key.txt')

maxlen=0
for x in X:
  if(len(x)>maxlen):
    maxlen=len(x)
print(maxlen)
print(X[0])
print(Y[0])
print(Z[0])

from transformers import BertTokenizer, BertModel
import torch

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Assuming `X` is your list of sentences
input_ids = []
attention_masks = []

print(f'Using device: {device}')

for sent in X:
    if not sent:
        continue
    encoded_dict = tokenizer.encode_plus(
        sent,                              # Sentence to encode
        add_special_tokens=False,           # Add '[CLS]' and '[SEP]'
        return_attention_mask=True,        # Construct attention masks
        return_tensors='pt',               # Return PyTorch tensors
        truncation=True
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0])

    input_ids.append(encoded_dict['input_ids'])

    attention_masks.append(encoded_dict['attention_mask'])

print('Original: ', X[0])
print('Token IDs:', input_ids[0])  # Call .cpu() before .numpy() if on GPU

model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True)
model.to(device)
model.eval()
embeddings = []

with torch.no_grad():
    for idx in range(len(input_ids)):
        input_id_batch = input_ids[idx].to(device)  # Add a batch dimension
        attention_mask_batch = attention_masks[idx].to(device)  # Add a batch dimension
        # print(input_id_batch.shape)
        output = model(input_ids=input_id_batch, attention_mask=attention_mask_batch)
        hidden_states = output[2]
        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings.size()
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings.size()
        token_embeddings = token_embeddings.permute(1,0,2)

        token_embeddings.size()
        token_vecs_sum = []


        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)

            token_vecs_sum.append(sum_vec)
        embeddings.append(token_vecs_sum)
#         print(len(X[0]),len(token_vecs_sum))
#         break

# Concatenate the list of embeddings along the first dimension to create a single tensor
# contextual_embeddings = torch.cat(embeddings, dim=0)

# Check the shape of the concatenated tensor
# print(embeddings.shape)

X_test,Y_test,Z_test = load_semcor_data('/kaggle/input/se-2013/semeval2013.data.xml', '/kaggle/input/se-2013/semeval2013.gold.key.txt')
input_ids_test = []
attention_masks_test = []

print(f'Using device: {device}')

# For every sentence...
for sent in X_test:
    if not sent:
        continue
    # `encode_plus` will perform multiple functions:
    encoded_dict = tokenizer.encode_plus(
        sent,                              # Sentence to encode
        add_special_tokens=False,           # Add '[CLS]' and '[SEP]'
        return_attention_mask=True,        # Construct attention masks
        return_tensors='pt',               # Return PyTorch tensors
        truncation=True
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0])

    # Add the encoded sentence to the list
    input_ids_test.append(encoded_dict['input_ids'])

    # And its attention mask
    attention_masks_test.append(encoded_dict['attention_mask'])

# Convert lists into tensors and move to the selected device
# input_ids = torch.cat(input_ids, dim=0).to(device)
# attention_masks = torch.cat(attention_masks, dim=0).to(device)

# Example: Print the first sentence's tokens and its corresponding IDs
print('Original: ', X_test[0])
print('Token IDs:', input_ids_test[0])  # Call .cpu() before .numpy() if on GPU

embeddings_test = []

with torch.no_grad():
    for idx in range(len(input_ids_test)):
        input_id_batch = input_ids_test[idx].to(device)  # Add a batch dimension
        attention_mask_batch = attention_masks_test[idx].to(device)  # Add a batch dimension
        # print(input_id_batch.shape)
#         output = model(input_ids=input_id_batch, attention_mask=attention_mask_batch)
        output = model(input_ids=input_id_batch, attention_mask=attention_mask_batch)
        hidden_states = output[2]
        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings.size()
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings.size()
        token_embeddings = token_embeddings.permute(1,0,2)

        token_embeddings.size()
        token_vecs_sum = []


        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)

            token_vecs_sum.append(sum_vec)
        embeddings_test.append(token_vecs_sum)

import numpy as np

contextual_sense_embeddings = {}

for sentence, sentence_embeddings, sentence_senses in zip(X, embeddings, Z):
    for word, word_embedding, sense in zip(sentence, sentence_embeddings, sentence_senses):
        if word not in contextual_sense_embeddings:
          contextual_sense_embeddings[word] = {}
        word_embedding = np.array(word_embedding.cpu().detach().numpy())
        if sense not in contextual_sense_embeddings[word]:
          contextual_sense_embeddings[word][sense] = []
        contextual_sense_embeddings[word][sense].append(word_embedding)
for word, senses_dict in contextual_sense_embeddings.items():
    for sense, embeddings_list in senses_dict.items():
        embeddings_array = np.array(embeddings_list)
        sense_embedding = np.mean(embeddings_array, axis=0)
        contextual_sense_embeddings[word][sense] = sense_embedding

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def get_nearest_sense(word,word_embedding):
    word_embedding_cpu = word_embedding.cpu()

    embedding_dict = contextual_sense_embeddings[word]
    sim = 0.0
    currsense = "test"
    for sense in embedding_dict:
        sense_embedding_np = np.array(embedding_dict[sense])
        word_embedding_np = np.array(word_embedding_cpu)

        # Compute cosine similarity
        curr_sim = cosine_similarity(word_embedding_np.reshape(1, -1), sense_embedding_np.reshape(1, -1))

        if curr_sim[0][0] > sim:
            sim = curr_sim[0][0]
            currsense = sense
    return currsense

total,correct,not_found=0,0,0
k=0
for i,sentence in enumerate(X_test):
    if not sentence:
        continue
    for j,word in enumerate(sentence):
        if(word in contextual_sense_embeddings):
#             print(k,j,word)
            predicted=get_nearest_sense(word,embeddings_test[k][j])
#             print(predicted,Z_test[i][j])
#             break
            if(predicted==Z_test[i][j]):
                correct+=1
        else:
            not_found+=1
        total+=1
    k+=1
print("total,correct,unknowns ", total,correct,not_found)

print(correct * 100/ (total - not_found))