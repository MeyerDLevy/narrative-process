import torch
from transformers import BertTokenizer, BertModel
from narrative_process.config import EMBEDDING_MODEL_NAME

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model globally using the config setting
tokenizer = BertTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = BertModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)

def load_custom_model(model_save_path=None):
    """
    Load a custom or pre-trained BERT model. Overrides the global model.
    """
    global model
    if model_save_path:
        model = BertModel.from_pretrained(model_save_path).to(device)
    else:
        model = BertModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)

def generate_bert_embeddings(sentence):
    """
    Generate BERT embeddings for a given sentence.
    Returns:
        - tokens_list: list of tokens
        - embeddings_list: list of numpy arrays (token embeddings, excluding [CLS] and [SEP])
    """
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    # Get token embeddings and move to CPU
    embeddings = outputs.last_hidden_state.squeeze().to('cpu')

    # Skip [CLS] and [SEP]
    tokens_list = tokens
    embeddings_list = [embedding.numpy() for embedding in embeddings[1:-1]]

    return tokens_list, embeddings_list
