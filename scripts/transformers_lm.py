import math
import os
from dataclasses import dataclass, field
from glob import glob
from typing import Any, Dict, Optional, Union

from torch.utils.data import ConcatDataset
from transformers import (CONFIG_MAPPING, MODEL_WITH_LM_HEAD_MAPPING,
                          AutoConfig, AutoModelWithLMHead, AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          DataCollatorForPermutationLanguageModeling,
                          LineByLineTextDataset, PretrainedConfig,
                          PreTrainedModel, PreTrainedTokenizer, TextDataset,
                          Trainer, TrainingArguments, set_seed)


class DataCollatorForWholeWordMask(object):
    ...


class LineByLineWithRefDataset(object):
    ...


try:
    from transformers import DataCollatorForWholeWordMask
except ImportError:
    pass
try:
    from transformers import LineByLineWithRefDataset
except ImportError:
    pass

MLM_MODEL_NAMES = ["bert", "roberta", "distilbert", "camembert"]
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())


@dataclass
class ModelArguments:
    """Arguments pertaining.

    :param model_name_or_path: The model checkpoint for weights initialization.
    :param model_type: If training from scratch, pass a model type.
    :param config_name: Pretrained config name or path if not the same as model_name.
    :param tokenizer_name: Pretrained tokenizer name or path if not the same as model_name.
    :param cache_dir: Where do you want to store the pretrained models downloaded from s3.
    """
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval.

    :param train_data_file: Path to the training file `(a single text file)`.
    :param train_data_files: Path to the training files directory; `(glob format; /*.txt)`.
        Very often splitting large files to smaller files can prevent tokenizer going out of memory.
    :param eval_data_file: An optional evaluation file to evaluate perplexity on `(a text file)`.
    :param chinese_ref_file: An optional input `ref data file` for whole word mask in Chinese.
    :param line_by_line: Whether distinct lines in the dataset are to be handled as distinct sequences.
    :param mlm: Train with masked-language modeling loss instead of language modeling.
    :param whole_word_mask: Whether or not to use whole word mask.
    :param mlm_probability: Ratio of tokens to mask for masked language modeling loss.
    :param plm_probability: Ratio of length of span of masked tokens to surrunding context length
        for permutation language modeling.
    :param max_span_length: Maximum length of a span of masked tokens for permutation language modeling.
    :param block_size: Optional input sequence length after tokenization. The training dataset will be
        truncated in block of this size for training. Default to the model max input length for single
        sentence inputs (take into account special tokens).
    :param overwrite_cache: Overwrite the cached training and evaluation sets.
    """
    train_data_file: Optional[str] = field(default=None)
    train_data_files: Optional[str] = field(default=None)
    eval_data_file: Optional[str] = field(default=None)
    chinese_ref_file: Optional[str] = field(default=None)
    line_by_line: bool = field(default=False)
    mlm: bool = field(default=False)
    whole_word_mask: bool = field(default=False)
    mlm_probability: float = field(default=1/6)
    max_span_length: int = field(default=5)
    block_size: int = field(default=-1)
    overwrite_cache: bool = field(default=False)


def select_dataloader(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
) -> Union[TextDataset, LineByLineTextDataset, ConcatDataset]:
    def dataloader(file_path):
        if args.line_by_line:
            if args.chinese_ref_file is not None:
                if not args.whole_word_mask or not args.mlm:
                    raise ValueError("You need to set world whole masking and mlm"
                                     " to True for Chinese Whole Word Mask")
                return LineByLineWithRefDataset(
                    tokenizer,
                    file_path,
                    args.block_size,
                    args.chinese_ref_file,
                )
            return LineByLineTextDataset(tokenizer, file_path, args.block_size)
        else:
            return TextDataset(
                tokenizer,
                file_path,
                args.block_size,
                args.overwrite_cache,
                cache_dir=cache_dir,
            )
    if evaluate:
        return dataloader(args.eval_data_file)
    elif args.train_data_files:
        return ConcatDataset([
            dataloader(fp) for fp in glob(args.train_data_files)
        ])
    else:
        return dataloader(args.train_data_file)


def train_language_model(
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments) -> Union[Dict[str, Any], None]:

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file."
        )
    if os.path.exists(training_args.output_dir) \
            and os.listdir(training_args.output_dir) \
            and training_args.do_train \
            and not training_args.overwrite_output_dir:
        raise ValueError(
            f"Output directory `{training_args.output_dir}` already exists"
            " and is not empty. Use --overwrite_output_dir to overcome."
        )

    set_seed(training_args.seed)

    # Configure pretrained configuration instance.
    config: PretrainedConfig = None
    cache_dir = model_args.cache_dir
    config_name = model_args.config_name
    model_name_or_path = model_args.model_name_or_path

    if config_name:
        config = AutoConfig.from_pretrained(config_name, cache_dir=cache_dir)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        print('You are instantiating a new config instance from scratch.')

    # Select and configure a pretrained model instance to use:
    model: PreTrainedModel = None
    if model_name_or_path:
        is_tensorflow_model = bool('.ckpt' in model_name_or_path)
        model = AutoModelWithLMHead.from_pretrained(
            model_name_or_path,
            from_tf=is_tensorflow_model,
            config=config,
            cache_dir=cache_dir,
        )
    else:
        model = AutoModelWithLMHead.from_config(config)
        print('Training new model from scratch.')

    # Select and configure a pretrained tokenizer instance to use:
    tokenizer = None
    if model_args.tokenizer_name:
        tokenizer = model_args.tokenizer_name
    elif model_name_or_path:
        tokenizer = model_name_or_path
    else:
        raise ValueError("Instantiating a new tokenizer is not supported")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir=cache_dir)
    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in MLM_MODEL_NAMES and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads."
            "They must be run using the mlm arg as True (masked language modeling).")

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Select and configure a dataloader to use for training and evaluation:
    train_dataset = None
    if training_args.do_train:
        train_dataset = select_dataloader(
            data_args, tokenizer, cache_dir=cache_dir)

    eval_dataset = None
    if training_args.do_eval:
        eval_dataset = select_dataloader(
            data_args, tokenizer, evaluate=True, cache_dir=cache_dir)

    # Select and configure a data-collator to use:
    data_collator = None
    if config.model_type == 'xlnet':
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    elif data_args.mlm and data_args.whole_word_mask:
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=data_args.mlm,
            mlm_probability=data_args.mlm_probability,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )
    model_path = None
    output_dir = training_args.output_dir
    if model_name_or_path is not None and os.path.isdir(model_name_or_path):
        model_path = model_name_or_path

    # Initialize training:
    if training_args.do_train:
        trainer.train(model_path=model_path)
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(output_dir)

    # Initialize evaluation:
    results = {}
    if training_args.do_eval:
        print('*** Evaluate ***')
        eval_output = trainer.evaluate()
        perplexity = math.exp(eval_output['eval_loss'])
        result = {'perplexity': perplexity}
        output_eval_fp = os.path.join(output_dir, 'eval_results_lm.txt')
        if trainer.is_world_master():
            with open(output_eval_fp, 'w') as writer:
                print('***** Eval Results *****')
                for key in sorted(result.keys()):
                    print(f'\t{key} = {result[key]}')
                    writer.write(f'{key} = {result[key]}\n')
        results.update(result)
    return results
