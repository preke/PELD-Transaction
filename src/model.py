from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel, RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertConfig, BertModel, BertPreTrainedModel
import torch.nn as nn
import torch.nn.functional as F
import torch
import re






# ======== BERT ========

class Dense(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()        
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print(x.shape)
        # x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class Emo_Generation(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 7
        self.bert = BertModel(config)
        self.mid_size = 768
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))

        self.mood_dense = Dense(self.mid_size+3, config.hidden_size, 3)
        self.mood_to_hidden = Dense(3, config.hidden_size, self.mid_size)
        self.mood_to_logit = Dense(3, config.hidden_size, 4)

        self.hidden_resize = Dense(config.hidden_size, config.hidden_size, self.mid_size)
        self.hidden_resize_2 = Dense(config.hidden_size, config.hidden_size, self.mid_size)

        self.personality_to_hidden = nn.Linear(3, self.mid_size)
        self.personality_dense = Dense(3, self.mid_size, 3)

        self.hidden_to_vad = Dense(config.hidden_size, config.hidden_size, 3)

        self.classifier = nn.Linear(self.mid_size*3, 7)

    def forward(self, input_ids, attn_masks, uttr_vad, personality, init_mood):
        
        bert_outputs   = self.bert(input_ids, attention_mask=attn_masks)
        bert_hidden    = bert_outputs[1]

        
        delta_mood = torch.cat((uttr_vad, self.hidden_resize(bert_hidden)), 1) 

        response_mood_vad    = F.softmax(self.mood_dense(delta_mood) * self.personality_dense(personality)) * self.scale + init_mood
        response_mood_logits = self.mood_to_logit(response_mood_vad)
        emo_embedding  = torch.cat((self.mood_to_hidden(torch.sign(response_mood_vad)), bert_hidden, self.personality_to_hidden(personality)), 1)
        response_emo   = self.classifier(emo_embedding)

        return response_mood_vad, response_mood_logits, response_emo


## mood and emotion distances
## moods inter distances

    
# class Emo_Generation(BertPreTrainedModel):

#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = 7
#         self.bert = BertModel(config)
#         self.mid_size = 768

#         self.mood_classifier = nn.Linear(self.mid_size, 4)
#         self.classifier = nn.Linear(self.mid_size, 7)

#     def forward(self, input_ids, attn_masks, uttr_vad, personality, init_mood):
        
#         bert_outputs   = self.bert(input_ids, attention_mask=attn_masks)
#         bert_hidden    = bert_outputs[1]
#         response_mood_logits = self.mood_classifier(bert_hidden)
#         response_emo   = self.classifier(bert_hidden)
#         response_mood_vad    = init_mood
#         return response_mood_vad, response_mood_logits, response_emo




   
# ======== RoBERTa ========

# class ClassificationHead(nn.Module):
    
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()  
#         self.dim_trans = nn.Linear(input_size, hidden_size)
#         self.dense = nn.Linear(hidden_size, hidden_size)
#         self.dropout = nn.Dropout(0.1)
#         self.out_proj = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.dim_trans(x) # 16 1 768
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x

    
    
# class Emo_Generation(RobertaPreTrainedModel):

#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = 7
#         self.mid_size = 100 
#         self.roberta = RobertaModel(config)
#         self.uttr_to_vad = nn.Linear(config.hidden_size, 3)
#         # self.classifier = ClassificationHead(3, config.hidden_size, 7)
#         self.classifier = nn.Linear(3, 7)

#     def forward(self, input_ids, attn_masks, uttr_vad, personality, init_mood):
        
#         roberta_outputs = self.roberta(input_ids, attention_mask=attn_masks)
#         roberta_hidden = roberta_outputs[0][:, 0, :]
        
#         response_mood = init_mood + uttr_vad
#         emo_embedding = self.uttr_to_vad(roberta_hidden) * personality + response_mood
#         response_emo = self.classifier(emo_embedding)
        
#         return response_emo, response_mood
    
    
