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
  
    - Path to phishing TXT file
    - Path to Glove embedding file.
  3) Run `prepare_dataset2.py` once again using the legit TXT file as the first argument.
 
## Training the model
  1) Run `train_autoencoder.py`with the tokenized phishing file as the argument to `input_data`
  2) The following hyperparameters can be assigned through command-line arguments:
  
    - `hidden_size`: LSTM Cell Size
    - `batch_size`: Batch Size during training
    - `n_epochs`: Number of training epochs
    - `seq_length`: Maximum number of words in a given input
  
## Evaluate the model
  1) Run `evaluate_autoencoder.py` with the tokenized phishing file as the argument to `input_data`
  2) Run `evaluate_autoencoder.py` once again with the tokenized legitimate file as the argument to `input_data`
  
# LSTM + Attention Based Phishing Detection (Experimental)

## Tokenizing the training data
Same as above

## Training and evaluating the model
  2) Run `train_lstm_attn_autoencoder.py` with same arguments as before. This script both builds the model and evaluates it.

