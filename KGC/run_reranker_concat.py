#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import csv
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import datasets
import jsonlines
import numpy as np
from datasets import load_dataset, load_metric, DatasetDict, Dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.data.data_collator import InputDataClass
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.21.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    codex_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to codex-s/m/l."}
    )
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to codex-s/m/l."}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    use_segment_ids: bool = field(
        default=True,
        metadata={"help": "If use token type ids when input to the model."},
    )
    add_kn: bool = field(
        default=False,
        metadata={"help": "If use token type ids when input to the model."},
    )
    negative_weight: float = field(
        default=0.5,
        metadata={"help": "If use token type ids when input to the model."},
    )
    top_1_only: bool = field(
        default=False,
        metadata={"help": "If use token type ids when input to the model."},
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    num_labels = 2

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    from modeling_bert import BertForSequenceClassificationWeight
    model = BertForSequenceClassificationWeight.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # Loading a dataset from your local files.
    def read_entity_or_relation(id_file, desc_file):
        outputs = {}
        with open(id_file, 'r') as f:
            for line in f.readlines():
                id = line.strip()
                outputs[id] = None
        with open(desc_file, 'r') as f:
            for line in f.readlines():
                id, desc = line.strip().split('\t')
                assert id in outputs
                outputs[id] = desc
        return outputs

    def read_triple_data(data_file, type='train'):
        _types, _heads, _relations, _tails, _labels, _external_kns, _tokenized_kns, _weights = [], [], [], [], [], [], [], []
        with jsonlines.open(data_file, 'r') as reader:
            for line in reader:
                triple = line['triple']
                label = int(line['label'])
                if line['kn'] and model_args.add_kn:
                    if type == 'train':
                        _heads.append(triple[0])
                        _relations.append(triple[1])
                        _tails.append(triple[2])
                        _labels.append(label)
                        _types.append(type)
                        if model_args.top_1_only:
                            kn = line['kn'][0]
                        else:
                            kn = tokenizer.sep_token.join(line['kn'])
                        _external_kns.append(kn)
                        _tokenized_kns.append(tokenizer.encode(kn, add_special_tokens=False))
                        if label == 1:
                            _weights.append(1.0)
                        elif label == 0:
                            _weights.append(model_args.negative_weight)
                        else:
                            raise Exception(f'UNKNOWN LABEL {label}')
                    else:
                        _heads.append(triple[0])
                        _relations.append(triple[1])
                        _tails.append(triple[2])
                        _labels.append(label)
                        _types.append(type)
                        if model_args.top_1_only:
                            kn = line['kn'][0]
                        else:
                            kn = tokenizer.sep_token.join(line['kn'])
                        _external_kns.append(kn)
                        _tokenized_kns.append(tokenizer.encode(kn, add_special_tokens=False))
                        _weights.append(1.0)
                else:
                    _heads.append(triple[0])
                    _relations.append(triple[1])
                    _tails.append(triple[2])
                    _labels.append(label)
                    _types.append(type)
                    _external_kns.append('')
                    _tokenized_kns.append([])
                    if label == 1:
                        _weights.append(1.)
                    elif label == 0:
                        _weights.append(model_args.negative_weight)
                    else:
                        raise Exception(f'UNKNOWN LABEL {label}')
        return {'type': _types, 'head': _heads, 'relation': _relations, 'tail': _tails, 'label': _labels,
                'kn': _external_kns, 'tokenized_kn': _tokenized_kns, 'weight': _weights}

    entities = read_entity_or_relation(os.path.join(data_args.codex_dir, 'entities.txt'),
                                       os.path.join(data_args.codex_dir, 'entity2text.txt'))
    relations = read_entity_or_relation(os.path.join(data_args.codex_dir, 'relations.txt'),
                                        os.path.join(data_args.codex_dir, 'relation2text.txt'))
    raw_train = read_triple_data(os.path.join(data_args.data_dir, 'train_sel.json'), 'train')
    raw_dev = read_triple_data(os.path.join(data_args.data_dir, 'valid_sel.json'), 'eval')
    raw_test = read_triple_data(os.path.join(data_args.data_dir, 'test_sel.json'), 'eval')

    raw_datasets = DatasetDict()
    raw_datasets['train'] = Dataset.from_dict(raw_train)
    raw_datasets['valid'] = Dataset.from_dict(raw_dev)
    raw_datasets['test'] = Dataset.from_dict(raw_test)

    # tokenize entities and relations
    tokenized_entities = {k: tokenizer.encode(v, add_special_tokens=False) for k, v in entities.items()}
    tokenized_relations = {k: tokenizer.encode(v, add_special_tokens=False) for k, v in relations.items()}
    # tokenize entities and relations (title only)
    _tokenized_entities = {k: tokenizer.encode(v.split(':')[0].strip(), add_special_tokens=False) for k, v in entities.items()}
    _tokenized_relations = {k: tokenizer.encode(v.split(':')[0].strip(), add_special_tokens=False) for k, v in relations.items()}
    entity_ids = {k for k in entities}
    gold_triples = {(x, y, z) for x, y, z in zip(raw_train['head'], raw_train['relation'], raw_train['tail'])}

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate([0, 1])}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "valid" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["valid"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    import torch

    def codex_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
        sources, segments, targets, weights = [], [], [], []
        for feature in features:
            if feature['type'] == 'train':
                # need negative
                head, tail, relation = feature['head'], feature['tail'], feature['relation']
                head_token_ids = [tokenizer.cls_token_id] + tokenized_entities[head] + [tokenizer.sep_token_id]
                relation_token_ids = tokenized_relations[relation] + [tokenizer.sep_token_id]
                tail_token_ids = tokenized_entities[tail] + [tokenizer.sep_token_id]
                input_token_ids = head_token_ids + relation_token_ids + tail_token_ids
                token_type_ids = [0] * len(head_token_ids) + [1] * len(relation_token_ids) + [0] * len(tail_token_ids)
                if feature['tokenized_kn'] and model_args.add_kn:
                    external_kn_token_ids = feature['tokenized_kn'] + [tokenizer.sep_token_id]
                    input_token_ids += external_kn_token_ids
                    token_type_ids += [1] * len(external_kn_token_ids)
                sources.append(input_token_ids)
                segments.append(token_type_ids)
                targets.append(feature['label'])
                weights.append(feature['weight'])

                # # add negative, dynamic select from entities
                # tmp_ent_list = random.sample(entity_ids, k=data_args.negative_num * 100)
                # negative_count = 0
                # for tmp_ent in tmp_ent_list:
                #     if negative_count >= data_args.negative_num:
                #         break
                #     rnd = random.random() # tail or head
                #     if rnd <= 0.5:
                #         # corrupting head
                #         if tmp_ent != head and (tmp_ent, relation, tail) not in gold_triples:
                #             tmp_head_token_ids = [tokenizer.cls_token_id] + tokenized_entities[tmp_ent] + [tokenizer.sep_token_id]
                #             input_token_ids = tmp_head_token_ids + relation_token_ids + tail_token_ids
                #             token_type_ids = [0] * len(tmp_head_token_ids) + [1] * len(relation_token_ids) + [0] * len(tail_token_ids)
                #             sources.append(input_token_ids)
                #             segments.append(token_type_ids)
                #             targets.append(0)
                #             negative_count += 1
                #             continue
                #     # corrupting tail
                #     elif tmp_ent != tail and (head, relation, tmp_ent) not in gold_triples:
                #         tmp_tail_token_ids = tokenized_entities[tmp_ent] + [tokenizer.sep_token_id]
                #         input_token_ids = head_token_ids + relation_token_ids + tmp_tail_token_ids
                #         token_type_ids = [0] * len(head_token_ids) + [1] * len(relation_token_ids) + [0] * len(tmp_tail_token_ids)
                #         sources.append(input_token_ids)
                #         segments.append(token_type_ids)
                #         targets.append(0)
                #         negative_count += 1
                #         continue
            else:
                head, tail, relation, label = feature['head'], feature['tail'], feature['relation'], feature['label']
                head_token_ids = [tokenizer.cls_token_id] + tokenized_entities[head] + [tokenizer.sep_token_id]
                relation_token_ids = tokenized_relations[relation] + [tokenizer.sep_token_id]
                tail_token_ids = tokenized_entities[tail] + [tokenizer.sep_token_id]
                input_token_ids = head_token_ids + relation_token_ids + tail_token_ids
                token_type_ids = [0] * len(head_token_ids) + [1] * len(relation_token_ids) + [0] * len(tail_token_ids)
                if feature['tokenized_kn'] and model_args.add_kn:
                    external_kn_token_ids = feature['tokenized_kn'] + [tokenizer.sep_token_id]
                    input_token_ids += external_kn_token_ids
                    token_type_ids += [1] * len(external_kn_token_ids)
                sources.append(input_token_ids)
                segments.append(token_type_ids)
                targets.append(label)
                weights.append(feature['weight'])

        batch = tokenizer.pad({'input_ids': sources, 'token_type_ids': segments, 'labels': targets},
                              padding='longest', return_tensors='pt', max_length=data_args.max_seq_length)
        batch['weights'] = torch.tensor(weights)
        if not model_args.use_segment_ids:
            batch.pop('token_type_ids')
        return batch

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=codex_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_datasets = [predict_dataset]

        for eval_dataset in predict_datasets:
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        for predict_dataset in predict_datasets:
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = [0, 1][item]
                        writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
