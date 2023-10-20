from argparse import ArgumentParser
from logging import getLogger
from json import load as json_load, dump as json_dump
from os.path import join as os_path_join
from sys import stdout as sys_stdout
from numpy import (
    append as np_append,
    argmax as np_argmax,
    squeeze as np_squeeze,
    concatenate as np_concatenate,
    exp as np_exp,
    array as np_array,
)
from tqdm import tqdm
from pandas import DataFrame
from torch import (
    no_grad as torch_no_grad,
    as_tensor as torch_as_tensor,
    int64 as torch_int64,
    float32 as torch_float32,
    load as torch_load,
    device as torch_device,
)
from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
# This line must be above local package reference
from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer,
                          RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification)

from utils.feature_extraction import (convert_examples_to_features, output_modes, processors)

ALL_MODELS = ('bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased', 'bert-base-multilingual-uncased', 'bert-base-multilingual-cased', 'bert-base-chinese', 'bert-base-german-cased', 'bert-large-uncased-whole-word-masking', 'bert-large-cased-whole-word-masking', 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-base-cased-finetuned-mrpc', 'bert-base-german-dbmdz-cased', 'bert-base-german-dbmdz-uncased', 'bert-base-japanese', 'bert-base-japanese-whole-word-masking', 'bert-base-japanese-char', 'bert-base-japanese-char-whole-word-masking', 'bert-base-finnish-cased-v1', 'bert-base-finnish-uncased-v1', 'roberta-base', 'roberta-large', 'roberta-large-mnli', 'distilroberta-base', 'roberta-base-openai-detector', 'roberta-large-openai-detector')
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

logger = getLogger(__name__)

@torch_no_grad()
def evaluate(args, model, tokenizer, device, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = args.task_name
    eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, device, evaluate=True)

    eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    len_eval_dataset = len(eval_dataset)
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d" % len_eval_dataset)
    print("  Batch size = %d" % eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    predictions = []
    ground_truth = []
    model.eval()
    use_segment_ids = args.model_type in ['bert', 'xlnet']
    for batch in tqdm(eval_dataloader, total=len_eval_dataset//eval_batch_size+1, desc='3.paragraph_ranking.evaluate', file=sys_stdout):
        labels = batch[3]
        inputs = {'input_ids':      batch[0].to(device),
                    'attention_mask': batch[1].to(device),
                    'token_type_ids': batch[2].to(device) if use_segment_ids else None,  # XLM don't use segment_ids
                    'labels': labels.to(device)        }
        del batch
        outputs = model(**inputs)
        del inputs
        tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.mean().item()

        logits = logits.detach().cpu().numpy()
        label_ids = labels.detach().cpu().numpy()

        predictions.append(logits)
        ground_truth.extend(list(label_ids))

        nb_eval_steps += 1
        if preds is None:
            preds = logits
            out_label_ids = label_ids
        else:
            preds = np_append(preds, logits, axis=0)
            out_label_ids = np_append(out_label_ids, label_ids, axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np_argmax(preds, axis=1)
    elif args.output_mode == "regression":
        preds = np_squeeze(preds)

    print("***** Writting Predictions ******")
    logits01 = np_concatenate(predictions, axis=0)
    logits0 = logits01[:, 0]
    logits1 = logits01[:, 1]
    score = DataFrame({'logits0': logits0, 'logits1': logits1, 'label': ground_truth})
    return score


def softmax(x):
    e_x = np_exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)


def rank_paras(data, pred_score):
    logits = np_array([pred_score['logits0'], pred_score['logits1']]).transpose()
    pred_score['prob'] = softmax(logits)[:, 1]

    ranked_paras = dict()
    cur_ptr = 0

    for case in tqdm(data):
        key = case['_id']
        tem_ptr = cur_ptr
        case_context = case['context']
        all_paras = []
        max_len = tem_ptr + len(case_context)
        while cur_ptr < max_len:
            score = pred_score.loc[cur_ptr, 'prob'].item()
            all_paras.append((case_context[cur_ptr - tem_ptr][0], score))
            cur_ptr += 1

        sorted_all_paras = sorted(all_paras, key=lambda x: x[1], reverse=True)
        ranked_paras[key] = sorted_all_paras

    return ranked_paras

def load_and_cache_examples(args, task, tokenizer, device, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]

    label_list = processor.get_labels()
    examples = processor.get_examples(args.input_data)
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    # Convert to Tensors and build dataset
    all_input_ids = torch_as_tensor([f.input_ids for f in features], dtype=torch_int64, device=device)
    all_input_mask = torch_as_tensor([f.input_mask for f in features], dtype=torch_int64, device=device)
    all_segment_ids = torch_as_tensor([f.segment_ids for f in features], dtype=torch_int64, device=device)
    if output_mode == "classification":
        all_label_ids = torch_as_tensor([f.label_id for f in features], dtype=torch_int64, device=device)
    elif output_mode == "regression":
        all_label_ids = torch_as_tensor([f.label_id for f in features], dtype=torch_float32, device=device)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset

def set_args():
    parser = ArgumentParser()

    ## Required parameters
    parser.add_argument("--eval_ckpt", default=None, type=str, required=True,
                        help="evaluation checkpoint")
    parser.add_argument("--raw_data", default=None, type=str, required=True,
                        help="raw data for processing")
    parser.add_argument("--input_data", default=None, type=str, required=True,
                        help="source data for processing")
    parser.add_argument("--data_dir", required=True, type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default='hotpotqa', type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--device_str", type=str, required=True, help="Device string. Can be either 'cuda', 'mps', or 'cpu'")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = set_args()

    # Setup CUDA
    device_str = args.device_str

    match device_str:
        case 'cuda':
            assert cuda_is_available()
            print('Using CUDA')
        case 'mps':
            assert mps_is_available()
            print('Using MPS')
        case 'cpu':
            print('Using CPU')
        case _:
            raise ValueError(f'Unknown device string {device_str}')
    device = torch_device(device_str)

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          device_map=device)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case, device_map=device)

    # Load a trained model that you have fine-tuned
    # model_state_dict = torch_load(f=args.eval_ckpt, map_location=device)
    # model = model_class.from_pretrained(args.model_name_or_path,
                                        # state_dict=model_state_dict,
    model = model_class.from_pretrained(args.eval_ckpt,
                                        config=config,
                                        device_map=device)
    score = evaluate(args, model, tokenizer, device, prefix="")

    # load source data
    with open(args.raw_data, 'r') as file_in:
        source_data = json_load(file_in)
    rank_paras_dict = rank_paras(source_data, score)
    with open(os_path_join(args.data_dir, 'para_ranking.json'), 'w') as file_out:
        json_dump(rank_paras_dict, file_out)