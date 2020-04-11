# LSTM-based Auto-encoders for phishing email detection

## Original GitHub repository
[Sentence autoencoder in Keras](https://github.com/basma-b/sentence_autoencoder_keras)

## Training Data

Please refer to [this link](https://github.com/shahriarhaque/themis#steps-to-visualize-data) to for instructions on pre-processing the [Nazario phishing dataset](https://monkey.org/~jose/phishing/phishing3.mbox) into a single TXT file.

## Download Glove Embeddings file

[Glove Embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip)

## Tokenizing the training data
  1) Edit `prepare_dataset2.py` and modify the `MODEL_DIRECTORY` and `TOKENIZER_FILE` variables as needed to direct the tokenized output.
  2) Run `prepare_dataset2.py` with two arguments:
    - Path to phishing/legit TXT file
    - Path to Glove embedding file.
 
## Training the model
  1) Run `train_autoencoder.py`
  2) The following hyperparameters can be assigned through command-line arguments:
    - `hidden_size`: LSTM Cell Size
    - `batch_size`: Batch Size during training
    - `n_epochs`: Number of training epochs
    - `seq_length`: Maximum number of words in a given input
  
## Evaluate the model
  1) Run `evaluate_autoencoder.py`
