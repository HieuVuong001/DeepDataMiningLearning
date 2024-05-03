import os
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering,
                          AutoTokenizer, pipeline, get_scheduler,
                          DataCollatorForSeq2Seq, DataCollatorWithPadding, MBartTokenizer, 
                          MBartTokenizerFast, default_data_collator, EvalPrediction)
import evaluate
import torch
from datasets import load_dataset, DatasetDict, get_dataset_split_names
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import math
from utils_qa import create_and_fill_np_array, postprocess_qa_predictions

trainoutput = './output'#"./output"
#taskname=args.traintag #taskname="eli5asksciencemodeling"
mycache_dir='./cache'

os.environ['TRANSFORMERS_CACHE'] = mycache_dir
os.environ['HF_HOME'] = mycache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(mycache_dir,"datasets") #"D:\Cache\huggingface\datasets" #os.path.join(hfcache_dir, 'datasets')
os.environ['HF_EVALUATE_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = "1"


model_checkpoint = 'distilbert/distilbert-base-uncased'
trainoutput = os.path.join(trainoutput, model_checkpoint, 'squad_v2'+'_'+ 'test')

os.makedirs(trainoutput, exist_ok=True)

def get_myoptimizer(model, learning_rate=2e-5, weight_decay=0.0):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer
def modelparameters(model, unfreezename=""):
    if unfreezename:
        for name, param in model.named_parameters():
            if name.startswith(unfreezename): # choose whatever you like here
                param.requires_grad = True
            else:
                param.requires_grad = False
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

def updateQAtraininputs(tokenized_examples, examples, tokenizer):
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    ## Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping") #new add

    #answers = examples["answers"]
    #start_positions = []
    #end_positions = []
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    #"overflow_to_sample_mapping"
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    # Looks like [0,1,2,2,2,3,4,5,5...] - Here 2nd input pair has been split in 3 parts

    #"offset_mapping"
    ## The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    # Looks like [[(0,0),(0,3),(3,4)...] ] - Contains the actual start indices and end indices for each word in the input.

    for i, offsets in enumerate(offset_mapping):#17 array, each array (offset) has 100 elements tuples of two integers representing the span of characters inside the original context.
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]

        cls_index = input_ids.index(tokenizer.cls_token_id)

        ## Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)  #[None 0 0... None 1 1 1... None] 100 tokens belongs to 0 or 1 or None
        # sequence_ids method to find which part of the offset corresponds to the question and which corresponds to the context.

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_idx = sample_mapping[i] #new add, get the index for samples
        #answer = answers[i]
        answers = examples["answers"][sample_idx] # sample_idx from sample_map
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            ## Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = answers["answer_start"][0] + len(answers["text"][0])
        
            #Option2
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    #inputs["start_positions"] = start_positions  #17 elements, if position is 0, means no answer in this region
    #inputs["end_positions"] = end_positions
    return tokenized_examples

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

            if mode=='train':
                #add "start_positions" and "end_positions" into the inputs as the labels
                model_inputs=updateQAtraininputs(model_inputs, examples, tokenizer)
            else: #val
                #add "example_id"
                model_inputs=updateQAvalinputs(model_inputs, examples)
            return model_inputs

def updateQAvalinputs(tokenized_examples, examples):
    # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")#100, if no overflow, then sample_map=0-99
    example_ids = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i) #[None, 0... None, 1... 1]

        ## One example can give several spans, this is the index of the example containing this span of text.
        sample_idx = sample_mapping[i]
        example_ids.append(examples["id"][sample_idx]) #example ids are strings

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        offset = tokenized_examples["offset_mapping"][i] #384 size array (0, 4)
        tokenized_examples["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ] #put None in sequence_id==1, i.e., put questions to None

    tokenized_examples["example_id"] = example_ids #string list
    return tokenized_examples

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


class myEvaluator:
    def __init__(self, useHFevaluator=False, dualevaluator=False):
        print("useHFevaluator:", useHFevaluator)
        print("dualevaluator:", dualevaluator)
        self.useHFevaluator = useHFevaluator
        self.dualevaluator = dualevaluator
        self.task = 'QA'
        self.metricname = "squad_v2"
        self.preds = []
        self.refs = []
        self.HFmetric = evaluate.load(self.metricname)

    def compute(self, predictions=None, references=None):
        if predictions is not None and references is not None:
            if self.useHFevaluator==True:
                results = self.HFmetric.compute(predictions=predictions, references=references)
                #keys: ['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
                print("HF evaluator:", results)
        else: #evaluate the whole dataset
            if self.useHFevaluator==True or self.dualevaluator==True:
                if self.task == "translation":
                    results = self.HFmetric.compute()
                elif self.task=="summarization":
                    results = self.HFmetric.compute(use_stemmer=True)
                #print("HF evaluator:", results["score"])
                print("HF evaluator:", results)
            
            if self.useHFevaluator==False or self.dualevaluator==True:
                if self.task=="translation":
                    #self.refs should be list of list strings
                    #Tokenization method to use for BLEU. If not provided, defaults to `zh` for Chinese, `ja-mecab` for Japanese, `ko-mecab` for Korean and `13a` (mteval) otherwise
                    if self.language=="zh":
                        bleu = sacrebleu.corpus_bleu(self.preds, [self.refs], tokenize="zh")
                    else:
                        bleu = sacrebleu.corpus_bleu(self.preds, [self.refs], tokenize="none")
                    results = {'score':bleu.score, 'counts':bleu.counts, 'totals': bleu.totals,
                            'precisions': bleu.precisions, 'bp': bleu.bp, 
                            'sys_len': bleu.sys_len, 'ref_len': bleu.ref_len
                            }
                elif self.task=="summarization":
                    results = self.localscorer._compute(self.preds, self.refs)
                print("Local evaluator:", results)
        
        return results
    
    def add_batch(self, predictions, references):
        if self.useHFevaluator==True or self.dualevaluator==True:
            self.HFmetric.add_batch(predictions=predictions, references=references)
        
        if self.useHFevaluator==False or self.dualevaluator==True:
            #self.preds.append(predictions)
            self.refs.extend(references)
            self.preds.extend(predictions)
            #references: list of list
            # for ref in references:
            #     self.refs.append(ref[0])
            #print(len(self.refs))

def evaluateQA_dataset(model, eval_dataloader, eval_dataset, raw_datasets, device, metric, trainoutput):
    # Evaluation
    totallen = len(eval_dataloader)
    print("Total evaluation length:", totallen)
    #evalprogress_bar = tqdm(range(num_training_steps))
    model.eval()
    all_start_logits = []
    all_end_logits = []
    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        #Get the highest probability from the model output for the start and end positions:
        all_start_logits.append(outputs.start_logits.cpu().numpy())
        all_end_logits.append(outputs.end_logits.cpu().numpy())
    
    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor: 384
    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len) #(5043, 384)
    end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len) #(5043, 384)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    predictions = (start_logits_concat, end_logits_concat)
    #prediction = post_processing_function(raw_datasets[valkey], eval_dataset, outputs_numpy)
    eval_examples = raw_datasets[valkey]
    # Post-processing: we match the start logits and end logits to answers in the original context.
    max_answer_length = 30
    n_best_size=20
    null_score_diff_threshold = 0.0
    stage="eval"
    predictions = predictions
    predictions = postprocess_qa_predictions(
        examples=eval_examples,
        features=eval_dataset,
        predictions=predictions,
        version_2_with_negative=True,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        null_score_diff_threshold=null_score_diff_threshold,
        output_dir=trainoutput,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if True:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    prediction = EvalPrediction(predictions=formatted_predictions, label_ids=references)

    result = metric.compute(prediction.predictions, prediction.label_ids)

    return result

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


# # Apply this QA preprocess function to every data instance in the dataset
train_dataset = train_dataset.map(
QApreprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names, fn_kwargs={"mode": mode})

mode='val'

eval_dataset =eval_dataset.map(
            QApreprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names,
                fn_kwargs={"mode": mode}) 

eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])

label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id

data_collator = default_data_collator

batch = data_collator([train_dataset[i] for i in range(1, 3)])


metric = myEvaluator(useHFevaluator=True, dualevaluator=False)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=data_collator,
    batch_size=64,
)

eval_dataloader = DataLoader(
            eval_dataset_for_model, collate_fn=data_collator, batch_size=64
        )

optimizer = get_myoptimizer(model, learning_rate=0.001)

num_train_epochs = 10
#num_update_steps_per_epoch = len(train_dataloader)
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / 2)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
completed_steps = starting_epoch * num_update_steps_per_epoch


lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
if torch.cuda.is_available():
    device = torch.device('cuda:'+str(0))  # CUDA GPU 0
    print(device)

model.to(device)

evaluateQA_dataset(model, eval_dataloader, eval_dataset, raw_datasets, device, metric, trainoutput)

print("Start training, total steps:", num_training_steps)
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(starting_epoch, num_train_epochs):
    # Training
    for step, batch in enumerate(train_dataloader):
        #batch = {k: v.to(device) for k, v in batch.items()}
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / 2

        loss.backward()

        if step % 2 == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

results = evaluateQA_dataset(model, eval_dataloader, eval_dataset, raw_datasets, device, metric, trainoutput)
