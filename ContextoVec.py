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

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = "This is a sample sentence, with punctuations! And some stop words."

# Tokenize the text into words
words = word_tokenize(text)

# Get list of English stopwords
stop_words = set(stopwords.words('english'))

# Remove punctuations
words = [word for word in words if word not in string.punctuation]

# Remove stop words
filtered_words = [word for word in words if word.lower() not in stop_words]

# Join the words back into a sentence
filtered_sentence = ' '.join(filtered_words)

print("Filtered sentence:", filtered_sentence)

# <sentence id="d000.s000">
# <wf lemma="how" pos="ADV">How</wf>
# <instance id="d000.s000.t000" lemma="long" pos="ADJ">long</instance>
# <wf lemma="have" pos="VERB">has</wf>
# <wf lemma="it" pos="PRON">it</wf>
# <instance id="d000.s000.t001" lemma="be" pos="VERB">been</instance>
# <wf lemma="since" pos="ADP">since</wf>
# <wf lemma="you" pos="PRON">you</wf>
# <instance id="d000.s000.t002" lemma="review" pos="VERB">reviewed</instance>
# <wf lemma="the" pos="DET">the</wf>
# <instance id="d000.s000.t003" lemma="objective" pos="NOUN">objectives</instance>
# <wf lemma="of" pos="ADP">of</wf>
# <wf lemma="you" pos="PRON">your</wf>
# <instance id="d000.s000.t004" lemma="benefit" pos="NOUN">benefit</instance>
# <wf lemma="and" pos="CONJ">and</wf>
# <instance id="d000.s000.t005" lemma="service" pos="NOUN">service</instance>
# <instance id="d000.s000.t006" lemma="program" pos="NOUN">program</instance>
# <wf lemma="?" pos=".">?</wf>
# </sentence>

import xml.etree.ElementTree as ET
import re

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

    sentences, pos, word_senses,word_sense_ids = [], [], [] ,[]
    for sentence in root.findall('.//sentence'):
        sentence_words, sentence_pos, sentence_ws, sentence_word_sense_ids = [], [],{},[]
        curr_word_sence_indices=[]
#         for word in sentence.findall('.//instance'):
        for word in sentence.findall('.//*') :
#             word_text = word.attrib.get('lemma', None)
            word_text=word.text.lower()
            word_pos = word.attrib.get('pos', None)
            word_id = word.attrib.get('id', None)
            word_sense = sense_keys.get(word_id, None)
            sentence_words.append(word_text)
#             sentence_pos.append(word_pos)
            if word_text is not None and word_pos is not None and word_id is not None:
                if word_sense:
                    sentence_ws[word_text]=word_sense[0]
#                     sentence_word_sense_ids.append(idx)
#         print(sentence_words)
        sentence_words = [re.sub(r'[^a-zA-Z\s-]', '', word) for word in sentence_words if word not in string.punctuation]
#         print(sentence_words)
        sentence_words = [word for word in sentence_words if (word not in stop_words or word in sentence_ws) and len(word)>0]
#         print(sentence_words)
        for idx,word in enumerate(sentence_words):
            if(word in sentence_ws):
                sentence_word_sense_ids.append(idx)
        sentences.append(sentence_words)
#         pos.append(sentence_pos)
        word_senses.append(sentence_ws)
        word_sense_ids.append(sentence_word_sense_ids)

    return sentences,word_senses,word_sense_ids

# Load SemCor data
X,Y,Z = load_semcor_data('/kaggle/input/wsd-training/semcor.data.xml', '/kaggle/input/wsd-training/semcor.gold.key.txt')

print(X[9])
print(Y[9])
print(Z[9])

def get_word2idx(sentences):
    word2idx={"</PAD>":0,"</UNK>":1}
    for words in sentences:
        for word in words:
            if word not in word2idx:
                word2idx[word]=len(word2idx)
    return word2idx

word2idx=get_word2idx(X)
print(len(word2idx))

train_X,train_Y=[],[]
for sentence in X:
    for i,word in enumerate(sentence):
#         forward,backward=sentence[0:i],sentence[i+1:]
        forward=[word2idx.get(word,1) for word in sentence[0:i]]
        backward=[word2idx.get(word,1) for word in sentence[i+1:][::-1]]
        train_X.append((forward,backward))
        train_Y.append(word2idx.get(sentence[i],1))

print(train_X[0:5],train_Y[0:5])

import torch
from torch.utils.data import Dataset
import random
import torch.nn.functional as F


class CVDataset(Dataset):
    def __init__(self, contexts, center_words, num_neg_samples, vocab_size, device):
        self.contexts = contexts
        self.center_words = center_words
        self.num_neg_samples = num_neg_samples
        self.vocab_size = vocab_size
        self.device = device

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        forward, backward = self.contexts[idx]
        target = self.center_words[idx]

        # Positive sample
        targets = [target]
        labels = [1]

        # Negative sampling
        neg_samples = []
        i = 0
        while i < self.num_neg_samples:
            negative_sample = torch.randint(0, self.vocab_size, (1,), device=self.device)
            if negative_sample != target and negative_sample not in forward and negative_sample not in backward:
                neg_samples.append(negative_sample.item())  # Convert tensor to int
                i += 1

        targets.extend(neg_samples)
        labels.extend([0] * len(neg_samples))

        # Shuffle targets and labels using the same permutation
        indices = list(range(len(labels)))
        random.shuffle(indices)
        targets = [targets[i] for i in indices]
        labels = [labels[i] for i in indices]

        # Convert targets and labels to tensors
        forward = torch.tensor(forward, device=self.device)
        backward = torch.tensor(backward, device=self.device)
        targets = torch.tensor(targets, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        pad_length_forward = max(0, 100 - len(forward))
        pad_length_backward = max(0, 100 - len(backward))
        forward = F.pad(forward, (0, pad_length_forward))
        backward = F.pad(backward, (0, pad_length_backward))
        forward = forward[:100]
        backward = backward[:100]
        return (forward, backward), targets, labels

dataset = CVDataset(train_X,train_Y,2,len(word2idx),device)

dataset[0]

from torch.utils.data import DataLoader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

import torch
import torch.nn as nn

class ELMO(nn.Module):
    def __init__(self, vocab_size, embedding_dim, batch_size, device):
        super(ELMO, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.device = device

        # Define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(self.device)

        # Define the forward and backward LSTMs
        self.forward_lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True, bidirectional=False).to(self.device)
        self.backward_lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True, bidirectional=False).to(self.device)

        # Define the linear layer
        self.linear = nn.Linear(embedding_dim * 2, embedding_dim).to(self.device)

    def forward(self, X):
        fwd, bwd, tgt = X  # Assuming X is a tuple of (forward_sequence, backward_sequence, target_word)

        # Move target tensor to device
        tgt = torch.tensor(tgt, dtype=torch.long, device=self.device)

        # Convert fwd, bwd, and tgt to tensors
        fwd = torch.tensor(fwd, dtype=torch.long, device=self.device)
        bwd = torch.tensor(bwd, dtype=torch.long, device=self.device)

        # Embed the target word
        word_emb = self.embedding(tgt)

        # Embed the forward and backward sequences
        fwd_emb = self.embedding(fwd)
        bwd_emb = self.embedding(bwd)

    # Rest of your forward pass...


        # Pass the embedded sequences through the forward and backward LSTMs
        fwd_out, _ = self.forward_lstm(fwd_emb)
        bwd_out, _ = self.backward_lstm(bwd_emb)

        # Select the last output from each LSTM
        fwd_last_out = fwd_out[:, -1, :]
        bwd_last_out = bwd_out[:, -1, :]

        # Concatenate the last outputs
        concatenated_rep = torch.cat((fwd_last_out, bwd_last_out), dim=1)

        # Apply linear transformation
        linear_out = self.linear(concatenated_rep)

        linear_out_broadcasted = linear_out.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]

        dots = linear_out_broadcasted * word_emb

        dots = dots.sum(dim=-1)

        return linear_out, dots

# Assuming you have imported the necessary modules and defined your CVDataset and ELMO classes
from tqdm import tqdm
# Define hyperparameters
vocab_size = len(word2idx)  # Example value, adjust based on your data
embedding_dim = 300  # Example value, adjust based on your data
batch_size = 32  # Example value, adjust based on your data
num_neg_samples = 2  # Example value, adjust based on your data
embedding_matrix = torch.randn(vocab_size, embedding_dim)  # Example value, use your pre-trained embeddings

# Create an instance of the dataset

# Create a DataLoader for batching and shuffling the data
# Instantiate the model
model = ELMO(vocab_size, embedding_dim, batch_size,device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
batch_count = 0
# Training loop
num_epochs = 5 # Example value, adjust based on your preference
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
        contexts, targets, labels = batch
        fwd_context,bwd_context=contexts
        # Zero the gradients
        optimizer.zero_grad()
#         print(fwd_context,bwd_context,targets)
        # Forward pass
#         outputs = model((fwd_context,bwd_context,targets))
#         break
        # Calculate the loss
        # Inside your training loop
        # Forward pass
        _,outputs = model((fwd_context, bwd_context, targets))
        # Apply softmax to convert outputs to class probabilities
#         class_probs = F.softmax(outputs, dim=1)
        # Calculate the loss using cross entropy loss
#         print(type(class_probs),outputs,class_probs)
        labels=labels.float()
#         print(type(labels),labels)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track total loss
        total_loss += loss.item()
        batch_count += 1

        # Print loss after every print_freq batches
        if batch_count % 100 == 0:
#             print(f"Epoch {epoch+1}, Batch {batch_count}, Loss: {total_loss / 100}")
            total_loss = 0.0  # Reset total_loss for the next print_freq batches

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(data_loader)}")

model.to(device)
model.eval()
embeddings = {}
with torch.no_grad():
    for i in range(len(X)):
        print((i + 1) * 100 / len(X))
        for ws_idx in Z[i]:
            current_word=X[i][ws_idx]
            current_word_sense=Y[i][current_word]
            if(current_word not in embeddings):
                embeddings[current_word]={}
            forward=[[word2idx.get(word,1) for word in X[i][0:i]]]
            backward=[[word2idx.get(word,1) for word in X[i][i+1:][::-1]]]
            forward = torch.tensor(forward, device=device)
            backward = torch.tensor(backward, device=device)
            target = torch.tensor(word2idx.get(current_word,1), device=device)
            pad_length_forward = max(0, 100 - len(forward))
            pad_length_backward = max(0, 100 - len(backward))
            forward = F.pad(forward, (0, pad_length_forward))
            backward = F.pad(backward, (0, pad_length_backward))
            forward = forward[:100]
            backward = backward[:100]
            if(current_word_sense not in embeddings[current_word]):
                embeddings[current_word][current_word_sense]=[]
            current_embedding,_=model((forward,backward,target))
            embeddings[current_word][current_word_sense].append(current_embedding[0])

# embeddings['long']
print("Hi")

import numpy as np
contextual_sense_embeddings={}
for word, senses_dict in embeddings.items():
    if(word not in contextual_sense_embeddings):
        contextual_sense_embeddings[word]={}
    for sense, embeddings_list in senses_dict.items():
#         if sense not in contextual_sense_embeddings[word]:
#             contextual_sense_embeddings[word][sense]=[]
# Convert a list of GPU tensors to a single NumPy array
        embeddings_array = np.array([embedding.cpu().numpy() for embedding in embeddings_list])
        sense_embedding = np.mean(embeddings_array, axis=0)
        contextual_sense_embeddings[word][sense] = sense_embedding
#         print(word,sense,sense_embedding)
#         break
#     break

# print(contextual_sense_embeddings['change-ringing'])

pip install nltk

print(nltk.data.path)

import nltk

# Download the WordNet corpora
nltk.download('wordnet')

import nltk
import subprocess

# Download and unzip wordnet
try:
    nltk.data.find('wordnet.zip')
except:
    nltk.download('wordnet', download_dir='/kaggle/working/')
    command = "unzip /kaggle/working/corpora/wordnet.zip -d /kaggle/working/corpora"
    subprocess.run(command.split())
    nltk.data.path.append('/kaggle/working/')

# Now you can import the NLTK resources as usual
from nltk.corpus import wordnet
# from collections import Counter

from nltk.corpus import wordnet as wn
print(wn.synsets('dog'))  # This should output a list of synsets associated

import numpy as np
from collections import Counter
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

def get_most_frequent_sense(word):
    # Look up synsets for the word in WordNet
    synsets = wn.synsets(word)

    if not synsets:
        return None  # No synsets found for the word

    # Collect all sense keys for the synsets of the word
    sense_keys = [sense.key() for synset in synsets for sense in synset.lemmas()]

    # You may need to adjust this part depending on how you obtain corpus statistics
    # Here, we simply count the occurrences of each sense key in a hypothetical corpus
    # Replace corpus_counts with your actual corpus statistics
    corpus_counts = Counter(sense_keys)

    # Find the most frequent sense key
    most_common_key = corpus_counts.most_common(1)[0][0]

    # Look up the synset associated with the most frequent sense key
    most_frequent_synset = wn.lemma_from_key(most_common_key).synset()

    return most_frequent_synset

# # Example usage
word = "apple"
most_frequent_sense = get_most_frequent_sense(word)
if most_frequent_sense:
    print(f"Most frequent sense of '{word}': {most_frequent_sense.name()}")
else:
    print(f"No sense found for '{word}' in WordNet")

X_test,Y_test,Z_test = load_semcor_data('/kaggle/input/se-2013/semeval2013.data.xml', '/kaggle/input/se-2013/semeval2013.gold.key.txt')
total,correct,not_found=0,0,0
k=0
mfs_correct=0

print(X_test[0],Y_test[0],Z_test[0])

with torch.no_grad():
    for i in range(len(X_test)):
        for ws_idx in Z_test[i]:
            current_word=X_test[i][ws_idx]
            current_word_sense=Y_test[i][current_word]
            forward=[[word2idx.get(word,1) for word in X_test[i][0:i]]]
            backward=[[word2idx.get(word,1) for word in X_test[i][i+1:][::-1]]]
            forward = torch.tensor(forward, device=device)
#             forward = forward.clone().detach().to(dtype=torch.long, device=self.device)
#             backward = backward.clone().detach().to(dtype=torch.long, device=self.device)

            backward = torch.tensor(backward, device=device)
            target = torch.tensor(word2idx.get(current_word,1), device=device)
            pad_length_forward = max(0, 100 - len(forward))
            pad_length_backward = max(0, 100 - len(backward))
            forward = F.pad(forward, (0, pad_length_forward))
            backward = F.pad(backward, (0, pad_length_backward))
            forward = forward[:100]
            backward = backward[:100]
            current_embedding,_=model((forward,backward,target))
#             print(current_embedding)
            if(current_word in contextual_sense_embeddings):
#                 print("If")
                predicted=get_nearest_sense(current_word,current_embedding)
#                 predicted=get_most_frequent_sense(current_word)
#                 print(predicted,current_word_sense)
                if(predicted==current_word_sense):
                    correct+=1
            else:
                not_found+=1
#             break
            total+=1
#         break

print(total,correct,not_found,mfs_correct)

print((correct + not_found) * 100 / (total - not_found))

import nltk
import subprocess

# Download and unzip wordnet
try:
    nltk.data.find('wordnet.zip')
except:
    nltk.download('wordnet', download_dir='/kaggle/working/')
    command = "unzip /kaggle/working/corpora/wordnet.zip -d /kaggle/working/corpora"
    subprocess.run(command.split())
    nltk.data.path.append('/kaggle/working/')


