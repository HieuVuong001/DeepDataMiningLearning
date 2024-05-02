import os
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering,
                          AutoTokenizer, pipeline, get_scheduler,
                          DataCollatorForSeq2Seq, DataCollatorWithPadding, MBartTokenizer, 
                          MBartTokenizerFast, default_data_collator, EvalPrediction)
import torch
from datasets import load_dataset, DatasetDict, get_dataset_split_names

trainoutput = './output'#"./output"
#taskname=args.traintag #taskname="eli5asksciencemodeling"
mycache_dir='./cache'

os.environ['TRANSFORMERS_CACHE'] = mycache_dir
os.environ['HF_HOME'] = mycache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(mycache_dir,"datasets") #"D:\Cache\huggingface\datasets" #os.path.join(hfcache_dir, 'datasets')
os.environ['HF_EVALUATE_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = "1"


model_checkpoint = 'distilbert-base-uncased'
trainoutput = os.path.join(trainoutput, model_checkpoint, 'squad_v2'+'_'+ 'test')

os.makedirs(trainoutput, exist_ok=True)

def modelparameters(model, unfreezename=""):
    if unfreezename:
        for name, param in model.named_parameters():
            if name.startswith(unfreezename): # choose whatever you like here
                param.requires_grad = True
            else:
                param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)


def QApreprocess_function(examples, mode='train'):
            # Break question into 
            questions = [ex.strip() for ex in examples[task_column]] #"question"
            context = examples[text_column] #"context"
            stride = 128
            model_inputs = tokenizer(
                questions,
                context, #examples["context"],
                max_length=384, #384
                truncation="only_second",
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True, #map the start and end positions of the answer to the original context 
                padding=padding, #"max_length",
            )

            print(model_inputs)
            # if mode=='train':
            #     #add "start_positions" and "end_positions" into the inputs as the labels
            #     model_inputs=updateQAtraininputs(model_inputs, examples, tokenizer)
            # else: #val
            #     #add "example_id"
            #     model_inputs=updateQAvalinputs(model_inputs, examples)
            return model_inputs


def loadmodel(model_checkpoint, task="QA", mycache_dir="", pretrained="", hpc=True, unfreezename=""):
        modelcache_dir=os.path.join(mycache_dir,'hub')
        #tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=modelcache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang="en", tgt_lang="zh", cache_dir=modelcache_dir)

        model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint, cache_dir=modelcache_dir)
        starting_epoch = 0
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')
            print("Pretrained epoch:", checkpoint['epoch'])
            starting_epoch = checkpoint['epoch'] +1
            model.load_state_dict(checkpoint['model_state_dict'])
        embedding_size = model.get_input_embeddings().weight.shape[0]
        print("Embeeding size:", embedding_size) #65001
        print("Tokenizer length:", len(tokenizer)) #65001
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        model_num_parameters = model.num_parameters() / 1_000_000
        print(f"'>>> Model number of parameters: {round(model_num_parameters)}M'")
        #print(f"'>>> BERT number of parameters: 110M'")
        modelparameters(model, unfreezename)

        return model, tokenizer, starting_epoch

def loaddata():
    task_column = ""
    text_column = ""
    target_column = ""

    raw_datasets = load_dataset('squad_v2')
    #raw_datasets = load_dataset("squad", split="train[:5000]") #'train', 'test'
    #raw_datasets["train"][0] #'id', 'title','context', 'question', 'answers' (dict with 'text' and 'answer_start'),  
    task_column ="question"
    text_column = "context"
    target_column = "answers"

    #Download to home/.cache/huggingface/dataset
    
    print("All keys in raw datasets:", raw_datasets['train'][0].keys())

    # Split the training data into training and testing
    # 90% training, 10% testing
    split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)

    raw_datasets = split_datasets
    
    # Take only a subset of the main dataset
    subset_num = 5000
    
    trainlen = int(min(subset_num, len(raw_datasets["train"])))
    testlen = int(trainlen/10)
    print("trainlen:", trainlen)
    print("testlen:", testlen)

    raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select([i for i in range(trainlen)])

    # Limit test size to 5000 max
    raw_datasets[valkey] = raw_datasets[valkey].shuffle(seed=42).select([i for i in range(min(testlen, 5000))])

    return raw_datasets, text_column, target_column, task_column

mycache_dir = './cache'
model, tokenizer, starting_epoch = loadmodel(model_checkpoint, task='QA', mycache_dir=mycache_dir, hpc=False, unfreezename='distilbert.transformer.layer.5')

valkey="test"

raw_datasets, text_column, target_column, task_column = loaddata()
column_names = raw_datasets["train"].column_names
padding = "max_length"
max_target_length = 128
ignore_pad_token_for_loss = True
stride = 128

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets[valkey]
mode='train'

examples = train_dataset[0]


toy = train_dataset.select(range(1))
print(toy)

toy.map(QApreprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names, fn_kwargs={"mode": mode})
# x = tokenizer(examples,
#             examples['context'], #examples["context"],
#             max_length=384, #384
#             truncation="only_second",
#             stride=stride,
#             return_overflowing_tokens=True,
#             return_offsets_mapping=True, #map the start and end positions of the answer to the original context 
#             padding=padding, #"max_length",
#             )

# print(tokenizer)
# # Apply this QA preprocess function to every data instance in the dataset
# train_dataset = train_dataset.map(
# QApreprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names, fn_kwargs={"mode": mode})
# print(column_names)