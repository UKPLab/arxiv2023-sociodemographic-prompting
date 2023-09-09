import os
import json
import numpy as np
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import hydra
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,\
    LlamaForCausalLM, T5ForConditionalGeneration
from api_client import OpenAIClient, run_api
from util import seq2seq_models, large_seq2seq_models, large_causal_models, sociodemographic_attribute_values, \
    label_maps, choices_per_dataset, dataset2prompt, compose_sociodemographic_info, format_prompt, fewshot_templates, \
    openai_models, binary_mapping, causal_models

# Parts of this code are adapted from [Ye et al. (2023)](https://arxiv.org/pdf/2302.05698.pdf)
# Original code: [https://github.com/HKUNLP/icl-ceil](https://github.com/HKUNLP/icl-ceil
# Modifications: We adapted the code such that it can work with different types of models and datasets


def parallel_run(func, args_list, n_processes=8, initializer=None, initargs=None, **kwargs):
    idx2res = {}
    func = partial(func, **kwargs)
    n = len(args_list)

    with Pool(n_processes, initializer=initializer, initargs=initargs) as p:
        for idx, response in tqdm(p.imap_unordered(partial(wrapper, func=func),
                                                   enumerate(args_list)),
                                  total=n):
            idx2res[idx] = response

    res = [idx2res[i] for i in range(n)]
    return res


def extract_response(response, mode='completion'):
    try:
        if mode == 'completion':
            return response['choices'][0]['text']
        elif mode == 'chat':
            return response['choices'][0]['message']['content']
        else:
            raise ValueError("mode must be one of 'completion' or 'chat'")
    except:
        return "error"


def wrapper(idx_args, func):
    idx, args = idx_args
    res = func(args)
    return idx, res


# Yield successive n-sized chunks from l.
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def ppl_generate(input_texts, model, tokenizer, choices, batch_size=8, device=None):
    result = []
    for batch in divide_chunks(input_texts, batch_size):
        loss_list = []
        # to support batch inference, here we assume the number of choices is equal for each instance
        for choices_list in list(zip(*([choices] * len(batch)))):
            filled_texts = []
            for text, choice in zip(batch, choices_list):
                filled_texts.append(text + " " + choice)
            loss_list.append(_evaluate_loss(filled_texts, model, tokenizer, device))
        lm_loss_list = np.array(loss_list)
        preds_ids = lm_loss_list.argmin(axis=0).tolist()
        preds = [choices[pred_id] for pred_id in preds_ids]  # turn ids into labels
        result.extend(preds)
    return result


def _evaluate_loss(input_texts, model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        if model.__class__ == LlamaForCausalLM:  # Llama model is not accepting token_type_ids
            inputs = {k: v.to(device) for k, v in inputs.items() if k != 'token_type_ids'}
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        if issubclass(model.__class__, T5ForConditionalGeneration):  # if using seq2seq models, we need to pass labels
            outputs = model(**inputs, labels=inputs['input_ids'])
            #outputs = model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
            #outputs.logits = torch.stack(list(outputs.scores), dim=1)  # generate() returns a tuple of tensors
        else:
            outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        # note here we assume padding is performed on the right, left padding token will affect position_id in gpt2
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
            shift_labels.size())
        ce_loss = loss.sum(-1).cpu().detach().numpy()  # -log(p(y))
        lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(-1).cpu().numpy()
    return ce_loss / lens


def compose_fewshot_examples(fewshot_samples, dataset, text_column, label_column, sociodemographic_info=True):

    label_conversion = label_maps[dataset]

    examples = []
    for sample in fewshot_samples:
        label = sample['ratings_majority']
        if dataset in ['toxicity_jigsaw', 'hatespeech_ghc']:  # binary tasks
            label = binary_mapping[label]
        format_atts = {'label': label, 'text': sample[text_column]}
        if dataset in ['stance_semeval2016t6', 'stance_gwsd']:
            format_atts['topic'] = sample['target']
        if sociodemographic_info:
            atts = {k: v for k, v in sample['ratings'].items() if k in sociodemographic_attribute_values.keys()}
            if 'age_range' in sample['ratings']:
                atts['age'] = sample['ratings']['age_range']
            if 'political_affilation' in sample['ratings']:
                atts['political_affiliation'] = sample['ratings']['political_affilation']
            template = fewshot_templates[dataset]['with_sd']
            template = template.format(sd_info=compose_sociodemographic_info(atts))
            format_atts.update(atts)
        else:
            template = fewshot_templates[dataset]['without_sd']
        example = template.format(**format_atts)
        examples.append(example)

    fewshot_examples = "\n".join(examples)
    return fewshot_examples


def get_dataset_prompt(cfg, text_column, label_column):
    dataset = cfg.task_name
    sociodemographic_info = cfg.sociodemographics
    sociodemographic_attributes = [cfg.sociodemographic_attribute]
    format_id = cfg.format_id
    order_id = cfg.order_id # currently not used, for later if we want to control for order of attributes

    fewshot_examples = ""
    if cfg.fewshot:
        with open(cfg.fewshot_file, 'r') as f:
            fewshot_samples_dict = json.load(f)
        fewshot_samples = fewshot_samples_dict[str(cfg.seed)][str(cfg.fewshot_k)][cfg.fewshot_strategy]
        fewshot_examples = compose_fewshot_examples(fewshot_samples, dataset, text_column,
                                                    label_column, sociodemographic_info)
        fewshot_examples += "\n"

    if sociodemographic_info:
        assert format_id < len(dataset2prompt[dataset]['with_sd'])
        prompt_template = dataset2prompt[dataset]['with_sd'][format_id]
        prompt = prompt_template.format(sd_info=compose_sociodemographic_info(sociodemographic_attributes))
    else:
        assert format_id <= len(dataset2prompt[dataset]['without_sd'])
        prompt = dataset2prompt[dataset]['without_sd'][format_id]

    return fewshot_examples + prompt


def read_data(cfg):

    in_fn = os.path.join(cfg.input_dir, cfg.input_file + ".jsonl")

    # data loading and preprocessing
    if cfg.task_name == 'toxicity_diverse_perspectives':
        text_column = 'comment'
        label_column = 'toxic_score'
    elif cfg.task_name == 'sentiment_diaz':
        text_column = 'unit_text'
        label_column = 'annotation'
    elif cfg.task_name == 'stance_semeval2016t6':
        text_column = 'text'
        label_column = 'stance'
    elif cfg.task_name == 'stance_gwsd':
        text_column = 'text'
        label_column = 'annotation'
    elif cfg.task_name == 'toxicity_jigsaw':
        text_column = 'comment_text'
        label_column = 'toxic'
    elif cfg.task_name == 'hatespeech_ghc':
        text_column = 'text'
        label_column = 'Hate'
    elif cfg.task_name == 'hatespeech_twitter':
        text_column = 'text'
        label_column = 'annotation'
    else:
        raise ValueError("dataset must be one of 'toxicity_diverse_perspectives', 'toxicity_jigsaw', "
                         "'stance_semeval2016t6', 'stance_gwsd' or 'sentiment_diaz'")
    inp = pd.read_json(in_fn, lines=True, orient='records')
    if cfg.sociodemographics:
        # filter duplicates based on text (without sociodemographics we run each example only once)
        inp = inp.drop_duplicates(subset=text_column)
        inp = inp.to_dict(orient='records')
        data = []
        for i in inp:
            # either we use the values for the specified sociodemographic attributes
            # or we use the existing ones in the data (ie ratings)
            if cfg.sociodemographic_attribute == 'ratings':
                for idx, rating in enumerate(i['ratings']):
                    row = {**i, **rating, 'rating_idx': idx}
                    data.append(row)
            else: # use all values of the specified sociodemographic attribute
                for val in sociodemographic_attribute_values[cfg.sociodemographic_attribute]:
                    row = {**i, cfg.sociodemographic_attribute: val}
                    data.append(row)
    else:
        data = inp.to_dict(orient='records')

    return data, text_column, label_column


def build_output_name(ds_name, cfg):
    out = ds_name
    # indicate sociodemographics
    if cfg.sociodemographics:
        out += "_with_sociodemographics_{att}"
    else:
        out += "_without_sociodemographics"
    # indicate model
    out += "_{model}"
    # indicate format
    out += "_format{format}"
    # indicate fewshot
    if cfg.fewshot:
        out += "_fewshot{fewshot}_strategy{strategy}_fewshotseed{fewshotseed}"
    return out


def parse_parameters(cfg):
    if cfg.model_name in openai_models:
        assert cfg.api_params.provider in ['openai', 'azure'], "Invalid provider for OpenAI models: " + cfg.api_params.provider
        assert cfg.api_params.api_key is not None, "OpenAI API key must be provided"
        if cfg.api_params.provider == 'azure':
            assert cfg.api_params.resource is not None, "Azure resource must be provided"
            assert cfg.api_params.deployment_name is not None, "Azure deployment name must be provided"
    else:
        assert os.path.exists(cfg.cache_dir), "Cache dir does not exist: " + cfg.cache_dir
    if cfg.fewshot:
        assert cfg.fewshot_file is not None, "Fewshot file must be provided"
        assert cfg.fewshot_k is not None, "Fewshot k must be provided"
        assert cfg.fewshot_strategy is not None, "Fewshot strategy must be provided"
        assert cfg.seed is not None, "Fewshot seed must be provided"
        fewshot_in_fn = os.path.join(cfg.input_dir, cfg.fewshot_file + ".json")
        cfg.fewshot_file = fewshot_in_fn  # set it for later use
        assert os.path.exists(fewshot_in_fn), "Fewshot file does not exist: " + fewshot_in_fn
        with open(fewshot_in_fn, 'r') as f:
            fewshot_samples_dict = json.load(f)
        assert str(cfg.seed) in fewshot_samples_dict, "Fewshot seed not found in fewshot file: " + str(cfg.seed)
        assert str(cfg.fewshot_k) in fewshot_samples_dict[str(cfg.seed)], "Fewshot k not found in fewshot file: " + str(cfg.fewshot_k)
        assert cfg.fewshot_strategy in fewshot_samples_dict[str(cfg.seed)][str(cfg.fewshot_k)], "Fewshot strategy not found in fewshot file: " + cfg.fewshot_strategy
        del fewshot_samples_dict
    if cfg.sociodemographics:
        assert cfg.sociodemographic_attribute is not None, "Sociodemographic attribute must be provided"
        #assert cfg.sociodemographic_attribute in [list(sociodemographic_attribute_values.keys()), 'ratings']


@hydra.main(config_path="configs", config_name="sd-prompting", version_base=None)
def main(cfg):
    parse_parameters(cfg)
    # TEST
    TEST = cfg.test
    ds_name = cfg.input_file
    cache_dir = cfg.cache_dir
    model_name = cfg.model_name
    model_name_for_path = model_name.replace("/", "-")

    # print experiment info
    print('Experiment: ', cfg.task_name)
    print('Model: ', model_name)
    print('Sociodemographics: ', cfg.sociodemographics)
    if cfg.sociodemographics:
        print('SD attributes: ', cfg.sociodemographic_attribute)

    data, text_column, label_column = read_data(cfg)

    # prompt templates
    name_features = {'model': model_name_for_path, 'format': cfg.format_id}
    if cfg.fewshot:
        name_features['fewshot'] = cfg.fewshot_k
        name_features['strategy'] = cfg.fewshot_strategy
        name_features['fewshotseed'] = cfg.seed

    if cfg.sociodemographics:
        att = cfg.sociodemographic_attribute
        name_features['att'] = att

    descriptor = build_output_name(ds_name, cfg)

    if TEST:
        data = data[:10]
        descriptor = descriptor + "_TEST"

    descriptor += ".jsonl"
    out_fn = os.path.join(cfg.output_dir, descriptor.format(**name_features))

    # load prompts
    prompt = get_dataset_prompt(cfg, text_column, label_column)
    args_list = [format_prompt(cfg, prompt, row, text_column) for row in data]

    if model_name in ["text-davinci-003"]:
        params = {**cfg.api_params, 'model': model_name}
        # initiate OpenAI client (either using openai directly or azure; see client code)
        client = OpenAIClient(**params)
        api_params = {'client': client, **cfg.api_params.generation_params}
        # API retrieval
        responses = parallel_run(run_api, args_list=args_list,
                                 n_processes=cfg.n_process,
                                 **api_params)

        # store data in jsonl format
        data_context = [{**row, 'prompt': prompt,
                         'response': extract_response(resp, mode=cfg.api_params.mode)} for row, prompt, resp
                        in zip(data, args_list, responses)]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name in large_causal_models:
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, load_in_8bit=True,
                                                         torch_dtype=torch.float16,
                                                         device_map='auto', local_files_only=True).eval()
        elif model_name in causal_models:
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True).eval()
            model.to(device)
        elif model_name in seq2seq_models:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True).eval()
            model.to(device)
        elif model_name in large_seq2seq_models:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True, cache_dir=cache_dir,
                                                          torch_dtype=torch.float16,
                                                          device_map='auto', local_files_only=True).eval()
        else:
            raise ValueError('No appropriate model found for model_name: ', model_name)

        # for local models, we do not parse the output into the label space
        # instead we prompt the model with each label value separately and take the one with minimal loss
        choices = choices_per_dataset[cfg.task_name]
        responses = ppl_generate(args_list, model, tokenizer, choices, cfg.generation_params.batch_size, device)

        # store data in jsonl format
        data_context = [{**row, 'prompt': prompt, 'response': resp} for row, prompt, resp in
                        zip(data, args_list, responses)]

    pd.DataFrame(data_context).to_json(out_fn, lines=True, orient='records')


if __name__ == "__main__":
    main()
