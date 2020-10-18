from pathlib import Path
from typing import Sequence, Tuple

from corona_nlp.dataset import CORD19Dataset
from corona_nlp.datatypes import Papers
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tqdm.auto import tqdm
from transformers import (DataCollatorForLanguageModeling,
                          LineByLineTextDataset, RobertaConfig,
                          RobertaForMaskedLM, RobertaTokenizerFast, Trainer,
                          TrainingArguments)

MODEL_NAME = "SciBERTa"
MODEL_PATH = "sciberta-lm-bpe/"
PAPERS_DIR = "sciberta-lm-bpe/papers/"
PAPERS_PKL = None

CORD19_DIR = "d:/Datasets/CORD-19-research-challenge/2020-03-13/"
NUM_PAPERS = -1  # -1 for all papers available in the directory.

SEQ_MAXLEN = 512
VOCAB_SIZE = 30522
MIN_TOK_FREQ = 2
MAX_POS_WX = 514
ATTN_HEADS = 12
NUM_HIDDEN = 6

BATCH_SIZE = 6
NUM_EPOCHS = 2
BLOCK_SIZE = 128
SAVE_STEPS = 10_000


def init_main_dirs() -> Tuple[Sequence[Path]]:
    roberta_root = Path(MODEL_NAME) if MODEL_PATH is None \
        else Path(MODEL_PATH).joinpath(MODEL_NAME)
    papers_data_dir = Path(PAPERS_DIR)
    training_files = roberta_root.joinpath("training_files/")
    training_dataset = roberta_root.joinpath("training_dataset/")
    checkpoints = roberta_root.joinpath("checkpoints/")
    directories = (roberta_root, papers_data_dir,
                   training_files, training_dataset, checkpoints)
    [
        p.mkdir(parents=True, exist_ok=True) for p in directories
    ]
    return directories


def papers_to_training_files(papers: Papers, out_dir: Path,
                             dataset: CORD19Dataset) -> None:
    ids = papers.indices
    sents = papers.sents
    joinpath = out_dir.joinpath
    for id in tqdm(ids, desc='paper-to-file'):
        fp = joinpath(f'{dataset[id]}.txt')
        with fp.open('w', encoding='utf-8') as file:
            for sent in sents(id):
                file.write(f'{sent}\n')
    print('Done saving {} sentences from {} total papers!'.format(
        papers.num_sents, papers.num_papers))


def papers_to_training_dataset(papers: Papers, fp: Path) -> None:
    ids = papers.indices
    sents = papers.sents
    with fp.open('w', encoding='utf-8') as file:
        for id in tqdm(ids, desc='line-by-line-dataset'):
            for sent in sents(id):
                file.write(f'{sent}\n')
    print('Done building the training dataset! lines: {}'.format(
        papers.num_sents))


def new_cord19_dataset(out_file: Path) -> Tuple[CORD19Dataset, Papers]:
    rootdir = Path(CORD19_DIR)
    sources = [p.joinpath(p.name) for p in rootdir.iterdir()
               if not p.name.endswith("4_17") and p.is_dir()]
    dataset = CORD19Dataset(
        source=sources,
        index_start=1,
        sort_first=True,
        nlp_model="en_core_sci_sm",
        text_keys=("body_text",),
    )
    sample = dataset.sample(NUM_PAPERS)
    papers = dataset.batch(sample, minlen=25)
    # Save the preprocessed papers to a pickle file before returning.
    filename = f"{papers.num_sents}_{papers.num_papers}_xl.pkl"
    papers.to_disk(out_file.joinpath(filename))
    return dataset, papers


def main():
    # Build the main directories for the model and training data.
    (roberta_root, papers_data_dir, training_files,
     training_dataset, checkpoints) = init_main_dirs()
    dataset: CORD19Dataset
    papers: Papers
    if PAPERS_PKL is None:
        dataset, papers = new_cord19_dataset(papers_data_dir)
    else:
        papers = Papers.from_disk(PAPERS_PKL)
        dataset = papers.init_cord19_dataset()

    # Build the data sources needed to train a new language model w/HF:
    training_dataset_file = training_dataset.joinpath("cord19.xl.txt")
    papers_to_training_dataset(papers, training_dataset_file)
    papers_to_training_files(papers, training_files, dataset=dataset)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_TOK_FREQ,
        files=[file.as_posix() for file in training_files.glob('*.txt')],
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )
    tokenizer.save_model(roberta_root.as_posix())

    tokenizer = ByteLevelBPETokenizer(
        roberta_root.joinpath("vocab.json").as_posix(),
        roberta_root.joinpath("merges.txt").as_posix(),
    )
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=SEQ_MAXLEN)

    config = RobertaConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POS_WX,
        num_attention_heads=ATTN_HEADS,
        num_hidden_layers=NUM_HIDDEN,
        type_vocab_size=1,
    )
    tokenizer = RobertaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=roberta_root.as_posix(),
        max_len=SEQ_MAXLEN,
    )
    model = RobertaForMaskedLM(config=config)

    line_by_line_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=training_dataset_file.as_posix(),
        block_size=BLOCK_SIZE,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15,
    )
    training_args = TrainingArguments(
        output_dir=checkpoints.as_posix(),
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=line_by_line_dataset,
        prediction_loss_only=True,
    )
    trainer.train()
    trainer.save_model(roberta_root.as_posix())
    print("\nDone training! {}ForMaskedLM model saved in path: {}\n".format(
        MODEL_NAME, roberta_root.absolute()))


if __name__ == '__main__':
    main()
