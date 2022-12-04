from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils import InputExample, InputFeatures
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer, ManualTemplate
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import torch
import pandas as pd

plm, tokenizer, model_config, WrapperClass = load_plm("gpt2","gpt2-large")


pokemon_descriptions = load_dataset('../data/dataset/', delimiter=';')
NUM_CLASSES = np.unique(pokemon_descriptions['train']['labels'])

split_pokemon_descriptions = pokemon_descriptions['train'].train_test_split(
    test_size=0.2, shuffle=True)

dataset = {}
for split in ['train']:
    dataset[split] = []
    for sample in split_pokemon_descriptions[split]:
        input_example = InputExample(text_a = sample['text'], label=int(sample['labels']))
        dataset[split].append(input_example)

promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} the pokemon is {"mask"}',
    tokenizer = tokenizer,
)

mappings = pd.read_csv('../data/pokemon_mapping.csv')
name_to_label_dict = mappings[["name","index"]].set_index('index').to_dict()["name"]

promptVerbalizer = ManualVerbalizer(
    classes = NUM_CLASSES,
    label_words = name_to_label_dict,
    tokenizer = tokenizer,
)

train_dataloader = PromptDataLoader(
  dataset=dataset["train"],
  template=promptTemplate, 
  tokenizer=tokenizer,
  tokenizer_wrapper_class=WrapperClass, 
  shuffle=True,
  truncate_method="head",
  decoder_max_length=3,
  batch_size=16,
  teacher_forcing=False,
  predict_eos_token=False,
  max_seq_length=512,
)

promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
    freeze_plm= False
)
promptModel= promptModel.cuda()

epochs = 10
no_decay = ['bias', 'layer_norm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        inputs = inputs.cuda()
        logits = promptModel(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()

        if step %100 == 0:
            print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

torch.save(promptModel.state_dict(),"checkp/gpt_2_large_trained_model.cp")

dataset_test = {}
for split in ['test']:
    dataset_test[split] = []
    for sample in split_pokemon_descriptions[split]:
        input_example = InputExample(text_a = sample['text'], label=int(sample['labels']))
        dataset_test[split].append(input_example)


test_dataloader = PromptDataLoader(dataset=dataset_test["test"], template=promptTemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
    batch_size=1,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

allpreds = []
alllabels = []
for step, inputs in enumerate(test_dataloader):
    inputs = inputs.cuda()
    logits = promptModel(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cuda().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cuda().tolist())

acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
print(acc)
