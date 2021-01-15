import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model (weights)
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

def encodeSentence(sentence):
    # Tokenize the sequence:
    tokens=tokenizer.tokenize(sentence)

    # Adding the cls and sep tokens to front and end
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    #print(" Tokens are \n {} ".format(tokens))

    # PAdding the tokens to fixed length T
    T=15
    padded_tokens=tokens +['[PAD]' for _ in range(T-len(tokens))]
    # Creating the attension mask
    attn_mask=[ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
    # Creating the seg_id
    seg_ids=[0 for _ in range(len(padded_tokens))]
    
    # Converting tokens into ids
    sent_ids=tokenizer.convert_tokens_to_ids(padded_tokens)
    
    # Creating torch tensors
    token_ids = torch.tensor(sent_ids).unsqueeze(0) 
    attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
    seg_ids   = torch.tensor(seg_ids).unsqueeze(0)

    # Encording sentence with ber model
    hidden_reps, cls_head = bert_model(token_ids, attention_mask = attn_mask,token_type_ids = seg_ids)
    # print(type(hidden_reps))
    # print(hidden_reps.shape ) #hidden states of each token in inout sequence 
    #print(cls_head.shape ) #hidden states of each [cls]
    # Try to using various embedding as sentence encording
    # encording = cls_head
    encording = cls_head.detach().cpu().numpy()
     
    return encording


