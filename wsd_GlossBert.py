!pip install transformers nltk

from google.colab import drive
drive.mount('/content/drive')

cd /content/drive/MyDrive/Nlp project/WSD_Unified_Evaluation_Datasets

import nltk
nltk.download('semcor')
from nltk.corpus import semcor
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import regex as re

from torch.utils.data import Dataset, DataLoader

semcor_tagged_sentences = semcor.tagged_sents(tag = 'wordnet')

print("Sample tagged sentence:", semcor_tagged_sentences[0])

from nltk.corpus import semcor
import nltk

# Ensuring that required resources are downloaded
nltk.download('semcor')
nltk.download('wordnet')

def extract_synsets(semcor_tagged_sentence):
    synsets = []
    for node in semcor_tagged_sentence:
        if isinstance(node, nltk.Tree):
            # Check if it's a Lemma node
            if isinstance(node.label(), nltk.corpus.reader.wordnet.Lemma):
                lemma = node.label()
                synset = lemma.synset()
                if synset:
                    synsets.append(synset.name())
            # Recursively check nested nodes
            synsets.extend(extract_synsets(node))
    return synsets

# Load the tagged sentences from SemCor
semcor_tagged_sentences = semcor.tagged_sents(tag='wordnet')

# Process sentences to extract synsets
sentence_synsets = []
for semcor_tagged_sentence in semcor_tagged_sentences:
    synsets = extract_synsets(semcor_tagged_sentence)
    sentence_synsets.append(synsets)

# Original sentences from SemCor
semcor_sentences = semcor.sents()
sentences = list(semcor_sentences)  # Convert to list for easier handling

# Diagnostic prints
print("Number of sentences:", len(sentences))
print("Number of tagged sentence synsets:", len(sentence_synsets))
if sentences:
    print("Sample sentence:", sentences[0])
if sentence_synsets:
    print("Sample sentence synsets:", sentence_synsets[0])

import string
from nltk.stem import WordNetLemmatizer

pip install nltk

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('semcor')

def remove_punctuation(sentence):
    translator = str.maketrans('', '', string.punctuation)
    sentence_no_punct = sentence.translate(translator)
    return sentence_no_punct

def preprocess_input(input_text):
    punctuation_removed_text=remove_punctuation(input_text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokenized_input = nltk.word_tokenize(punctuation_removed_text)
    filtered_tokens = [token.lower() for token in tokenized_input if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

from nltk.corpus import wordnet

def get_word_senses(word):
    synsets = wordnet.synsets(word)
    senses = []
    for synset in synsets:
        senses.append(synset.definition())
    return senses

# Creating context-gloss pairs
def create_gloss_sense_pairs(sentences, sentences_synsets):
    gloss_pairs = []
    for sentence, sentence_synset in zip(sentences, sentences_synsets):
        preprocessed_sentence = preprocess_input(' '.join(sentence))
        for synset_id in sentence_synset:
            if '..' in synset_id:
                continue
            sense_name = synset_id.split('.')[0]
            preprocessed_sense_name = preprocess_input(sense_name)
            if not preprocessed_sense_name:
                continue
            for i, sense in enumerate(wordnet.synsets(sense_name)):
                preprocessed_sense = preprocess_input(sense.definition())
                if preprocessed_sense_name in preprocessed_sentence:
                    # Assume initially all pairs are incorrect (label 0)
                    label = 0
                    # If the sense index matches the sense number from SemCor, label it as correct (label 1)
                    if i + 1 == int(synset_id.split('.')[2]):
                        label = 1
                    gloss_pairs.append((preprocessed_sentence, f"<target> {preprocessed_sense_name} </target>: {preprocessed_sense}", preprocessed_sense_name, label))
    return gloss_pairs

gloss_pairs = create_gloss_sense_pairs(semcor_sentences, sentence_synsets)

# Create the DataFrame
df = pd.DataFrame(gloss_pairs, columns=['Sentence', 'Sense_definition', 'Sense_Name', 'Label'])

from nltk.corpus import wordnet as wn



gloss_pairs = create_gloss_sense_pairs(sentences, sentence_synsets)

print(gloss_pairs[1])



import pandas as pd

rows = []

# Function to add data to the DataFrame
def add_data(sentence, sense, sense_names, label):
    new_row = {'Sentence': sentence, 'Sense_definition': sense, 'Sense_Names': sense_names, 'Label': label}
    rows.append(new_row)

# Example of adding data
for i in range(0, len(gloss_pairs)):
    sentence, sense_definition, sense_name, label = gloss_pairs[i]
    add_data(sentence, sense_definition, sense_name, label)

df = pd.DataFrame(rows)

print("Number of gloss pairs:", len(gloss_pairs))
if gloss_pairs:
    print("Sample gloss pair:", gloss_pairs[0])



class GlossBERT(nn.Module):
    def __init__(self, bert_model):
        super(GlossBERT, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)  # Binary classification

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_token_embedding = outputs.last_hidden_state[:, 0, :]  # Get [CLS] token embedding
        logits = self.fc(cls_token_embedding)
        return logits

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
gloss_bert_model = GlossBERT(bert_model)

class GlossDataset(Dataset):
    def __init__(self, dataframe, include_labels=False):
        self.dataframe = dataframe
        self.include_labels = include_labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        context = item['Sentence']
        gloss = item['Sense_definition']
        if self.include_labels:
            label = item['Label']
            return context, gloss, label
        return context, gloss

# Prepare the dataset and dataloader
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = GlossDataset(train_df)
test_dataset = GlossDataset(test_df)
train_loader = DataLoader(GlossDataset(train_df, include_labels=True), batch_size=16, shuffle=True)
test_loader = DataLoader(GlossDataset(test_df, include_labels=False), batch_size=16, shuffle=False)

test_loader = DataLoader(GlossDataset(test_df, include_labels=True), batch_size=16, shuffle=False)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gloss_bert_model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(gloss_bert_model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

def calculate_accuracy(preds, y):
    """
    Calculates accuracy per batch.
    preds: Tensor of predictions
    y: Tensor of true labels
    """
    _, predicted_labels = torch.max(preds, 1)  # Get the index of the max log-probability
    correct = (predicted_labels == y).float()  # Convert correct matches to float and then compute the mean
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for context, gloss, label in iterator:
        optimizer.zero_grad()
        inputs = tokenizer(context, gloss, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = torch.tensor(label).to(device)

        outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask)
        loss = criterion(outputs, labels)

        acc = calculate_accuracy(outputs, labels)  # Calculate accuracy

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    example_pairs = {}
    total_correct = 0
    total = 0

    with torch.no_grad():
        for context, gloss, labels in iterator:
            labels = torch.tensor(labels).to(device)  # Ensure labels are on the correct device

            inputs = tokenizer(context, gloss, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask)
            probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            predictions = torch.argmax(probabilities, dim=1)  # Prediction based on highest probability

            # Collect examples for a specific context and target word
            for i in range(len(context)):
                key = (context[i], gloss[i].split(" ")[1])  # Assumes target word is always second word in gloss
                if key not in example_pairs:
                    example_pairs[key] = {"positive": None, "negative": None}

                label_type = "positive" if labels[i] == 1 else "negative"
                if example_pairs[key][label_type] is None or (predictions[i] == labels[i]):
                    example_pairs[key][label_type] = (context[i], gloss[i], predictions[i], labels[i], predictions[i] == labels[i])

            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # Display examples with both positive and negative cases for the same context and target word
        for key, types in example_pairs.items():
            if types["positive"] and types["negative"]:
                print(f"\nContext: {key[0]}")
                print(f"Target Word: {key[1]}")
                for label_type, details in types.items():
                    if details:
                        print(f"\nExample of {label_type} case:")
                        print(f"Gloss: {details[1]}")
                        print(f"Prediction: {details[2]}, Actual Label: {details[3]}, Correct: {details[4]}")
                print("----")

    accuracy = total_correct / total if total > 0 else 0
    return accuracy



# Assuming the model was saved with the filename 'GlossBERT_model_1.pth'
model_path = 'GlossBERT_model_1.pth'

# Load the model weights
gloss_bert_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

gloss_bert_model.eval()

num_epochs = 1
for epoch in range(num_epochs):
    train_loss, train_acc = train(gloss_bert_model, train_loader, optimizer, criterion)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gloss_bert_model.to(device)

test_data = GlossDataset(test_df, include_labels=False)
print(test_data[0])
print(test_data[1])

# Assuming 'model', 'test_loader', and 'criterion' are already defined
accuracy = evaluate(gloss_bert_model, test_loader, criterion)  # Shows up to 3 examples of each type
print(f"Test Accuracy: {accuracy:.4f}")

def parse_semeval_data(xml_path, gold_key_path):
    # Load the gold keys
    gold_keys = {}
    with open(gold_key_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            word_id = parts[0]
            senses = parts[1:]
            gold_keys[word_id] = senses

    # Parse the XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []

    for sentence in root.findall('.//sentence'):
        # Build the sentence text from parts
        text_parts = []
        for element in sentence:
            if element.tag == 'wf' or element.tag == 'instance':
                text_parts.append(element.text)
        sentence_text = ' '.join(text_parts)

        # Collect instances and their context
        for instance in sentence.findall('.//instance'):
            instance_id = instance.get('id')
            lemma = instance.get('lemma')
            if instance_id in gold_keys:
                # Only consider the first sense (common practice)
                correct_sense = gold_keys[instance_id][0]
                data.append((sentence_text, instance_id, lemma, correct_sense))
    return data

semeval_data = parse_semeval_data('semeval2015/semeval2015.data.xml', 'semeval2015/semeval2015.gold.key.txt')

def create_gloss_sense_pairs_for_semeval(data):
    gloss_pairs = []
    for sentence, instance_id, lemma, correct_sense in data:
        preprocessed_sentence = preprocess_input(sentence)
        synsets = wn.synsets(lemma)
        # Create a set of all valid sense keys for the lemma
        valid_sense_keys = {syn.lemmas()[0].key().split('%')[1] for syn in synsets}
        correct_sense_key = correct_sense.split('%')[1]  # Assuming the key format matches

        for synset in synsets:
            sense_key = synset.lemmas()[0].key().split('%')[1]
            gloss = synset.definition()
            preprocessed_gloss = preprocess_input(gloss)
            # Label as positive if the sense key matches, otherwise negative
            label = 1 if sense_key == correct_sense_key else 0
            gloss_pairs.append((preprocessed_sentence, f"<target> {lemma} </target>: {preprocessed_gloss}", lemma, label))

    return gloss_pairs

try:
    semeval_data = parse_semeval_data('semeval2015/semeval2015.data.xml', 'semeval2015/semeval2015.gold.key.txt')
    print("Data parsed successfully. Number of entries:", len(semeval_data))
    # Continue with creating gloss pairs and evaluating your model
    test_pairs = create_gloss_sense_pairs_for_semeval(semeval_data)
    # Assume test_pairs is now ready to be loaded into your evaluation framework
    print("Ready to evaluate")
except Exception as e:
    print("Error parsing the data:", str(e))

df = pd.DataFrame(test_pairs, columns=['Sentence', 'Sense_definition', 'Sense_Name', 'Label'])

from torch.utils.data import Dataset, DataLoader
import torch

class GlossDataset(Dataset):
    def __init__(self, dataframe, include_labels=True):
        self.dataframe = dataframe
        self.include_labels = include_labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        context = item['Sentence']
        gloss = item['Sense_definition']
        if self.include_labels:
            label = item['Label']
            return context, gloss, label
        return context, gloss

import torch.nn.functional as F

def evaluate_probability_based(model, data_loader, device):
    model.eval()
    results = {}
    final_predictions = {}

    with torch.no_grad():
        for context, gloss, labels in data_loader:
            inputs = tokenizer(context, gloss, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask)
            probabilities = F.softmax(outputs, dim=1)[:, 1]  # Assuming index 1 corresponds to "yes"

            for i, (ctx, gls, prob, label) in enumerate(zip(context, gloss, probabilities, labels)):
                key = (ctx, gls.split("<target>")[1].split("</target>")[0].strip())  # More precise target extraction
                if ctx not in final_predictions:
                    final_predictions[ctx] = {'gloss': None, 'highest_prob': -1, 'label': None}

                # Check if this gloss's probability is the highest seen so far for this context
                if prob.item() > final_predictions[ctx]['highest_prob']:
                    final_predictions[ctx]['highest_prob'] = prob.item()
                    final_predictions[ctx]['gloss'] = gls
                    final_predictions[ctx]['label'] = label.item()

        # Determine overall accuracy
        total_correct = sum(1 for k, v in final_predictions.items() if v['label'] == 1)  # Assuming label 1 is the correct prediction
        total = len(final_predictions)

    accuracy = total_correct / total if total > 0 else 0
    return accuracy

# Assume df is your DataFrame containing the test data
test_dataset = GlossDataset(df, include_labels=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Assume model is your GlossBERT model and it's already loaded
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gloss_bert_model.to(device)

# Evaluate the model
model_accuracy = evaluate_probability_based(gloss_bert_model, test_loader, device)
print(f"Model accuracy on the test data: {model_accuracy:.4f}")

|