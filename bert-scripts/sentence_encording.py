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
    #1.Tokenize the sequence:
    tokens=tokenizer.tokenize(sentence)
    print(tokens)
    print(type(tokens))

    tokens = ['[CLS]'] + tokens + ['[SEP]']
    print(" Tokens are \n {} ".format(tokens))

    T=15
    padded_tokens=tokens +['[PAD]' for _ in range(T-len(tokens))]
    print("Padded tokens are \n {} ".format(padded_tokens))
    attn_mask=[ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
    print("Attention Mask are \n {} ".format(attn_mask))

    seg_ids=[0 for _ in range(len(padded_tokens))]
    print("Segment Tokens are \n {}".format(seg_ids))

    sent_ids=tokenizer.convert_tokens_to_ids(padded_tokens)
    print("senetence idexes \n {} ".format(sent_ids))
    token_ids = torch.tensor(sent_ids).unsqueeze(0) 
    attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
    seg_ids   = torch.tensor(seg_ids).unsqueeze(0)

    hidden_reps, cls_head = bert_model(token_ids, attention_mask = attn_mask,token_type_ids = seg_ids)
    print(type(hidden_reps))
    # print(hidden_reps.shape ) #hidden states of each token in inout sequence 
    print(cls_head.shape ) #hidden states of each [cls]
    return cls_head


# sentence='I really enjoyed this movie a lot.'
# #1.Tokenize the sequence:
# tokens=tokenizer.tokenize(sentence)
# print(tokens)
# print(type(tokens))

# tokens = ['[CLS]'] + tokens + ['[SEP]']
# print(" Tokens are \n {} ".format(tokens))

# T=15
# padded_tokens=tokens +['[PAD]' for _ in range(T-len(tokens))]
# print("Padded tokens are \n {} ".format(padded_tokens))
# attn_mask=[ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
# print("Attention Mask are \n {} ".format(attn_mask))

# seg_ids=[0 for _ in range(len(padded_tokens))]
# print("Segment Tokens are \n {}".format(seg_ids))

# sent_ids=tokenizer.convert_tokens_to_ids(padded_tokens)
# print("senetence idexes \n {} ".format(sent_ids))
# token_ids = torch.tensor(sent_ids).unsqueeze(0) 
# attn_mask = torch.tensor(attn_mask).unsqueeze(0) 
# seg_ids   = torch.tensor(seg_ids).unsqueeze(0)

# hidden_reps, cls_head = bert_model(token_ids, attention_mask = attn_mask,token_type_ids = seg_ids)
# print(type(hidden_reps))
# # print(hidden_reps.shape ) #hidden states of each token in inout sequence 
# print(cls_head.shape ) #hidden states of each [cls]
