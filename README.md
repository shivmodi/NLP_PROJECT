# Word Sense Disambiguation (WSD) Using BERT-based Models

This project focuses on **Word Sense Disambiguation (WSD)**, which is a crucial task in Natural Language Processing (NLP) aimed at identifying which sense of a word is used in a given sentence, especially when the word has multiple meanings. We explore three different models for WSD, all based on **BERT** and its variants, leveraging contextual embeddings to improve accuracy.

## Models Used

### 1. **Nearest Sense**
- **Approach**: This model uses **pre-trained BERT embeddings** and computes **cosine similarity** to match the target word’s contextual embedding with precomputed sense embeddings.
- **Key Components**:
  - Pre-trained **BERT** model for generating contextual embeddings.
  - **Cosine similarity** to find the most similar sense.
  - **Training**: Uses labeled data (SemCor) to train and compute sense embeddings.
  - **Testing**: For a new sentence, it computes the contextual embedding of the target word and selects the most similar sense embedding.

### 2. **Context2Vec**
- **Approach**: **Context2Vec** uses **bidirectional LSTMs (BiLSTM)** to capture the surrounding context of a target word. It computes contextualized word embeddings by considering both the left and right context of a word.
- **Key Components**:
  - **BiLSTM** for capturing rich contextual information.
  - **Word embeddings** for each word in the sentence.
  - **Training**: LSTM-based network is trained using **context windows** around the target word.
  - **Testing**: Predicts the word sense by comparing the target word's contextual embedding with learned sense embeddings.

### 3. **GlossBERT**
- **Approach**: **GlossBERT** extends BERT by integrating **word glosses (short definitions)** from WordNet into the model, treating WSD as a **sentence pair classification** task. It predicts whether the sense defined by the gloss matches the context of the word.
- **Key Components**:
  - **Pre-trained BERT** for generating contextual embeddings.
  - **WordNet glosses** for each word sense.
  - **Fine-tuning** on task-specific data to learn the association between context and gloss.
  - **Testing**: The model predicts the correct word sense by classifying the match between the word’s context and its gloss.

---

## Dataset

We used the **SemCor** and **SemEval** datasets for training and evaluation. These datasets provide:
- **Semantically annotated sentences** with **WordNet sense IDs**, making them ideal for WSD tasks.
- **Two key files**:
  - `semcor.data`: Contains the sentences and their corresponding sense ID information.
  - `semcor.gold.key`: Maps the sense IDs to WordNet sense annotations.

---

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:shivmodi/NLP_PROJECT.git
   cd NLP_PROJECT
