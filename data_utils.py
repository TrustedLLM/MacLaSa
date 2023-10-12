import os
import json
import math
import random
import logging

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def build_dataload_and_cache_examples(args, tokenizers, evaluate=False):
    assert isinstance(tokenizers, list)
    args.batch_size = (
        args.per_gpu_train_batch_size if not evaluate else args.per_gpu_eval_batch_size
    ) * max(1, args.n_gpu)
    senti_file_path = (
        args.train_senti_data_file if not evaluate else args.eval_senti_data_file
    )
    topic_file_path = (
        args.train_topic_data_file if not evaluate else args.eval_topic_data_file
    )

    dataset = BucketBatchDataset(
        senti_file_path, topic_file_path, args.batch_size, tokenizers
    )
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=True,
        collate_fn=BucketBatchDataset.collate_fn,
        sampler=sampler,
    )
    return dataloader


class TokenDataset(Dataset):
    def __init__(
        self,
        tokenizers,
        file_path="train",
        block_size=512,
    ):
        print("loading from file: ", file_path)
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_" + f"{filename}.json",
        )

        self.examples = []
        self.tokenizers = tokenizers

        self.new_version = "fast" in tokenizers[0].__doc__

        # Bert tokenizer special tokens
        self.bert_pad_token = tokenizers[0].convert_tokens_to_ids(
            [tokenizers[0].pad_token]
        )[0]

        # GPT-2 tokenizer special tokens
        self.gpt2_pad_token = tokenizers[1].convert_tokens_to_ids(
            [tokenizers[1].pad_token]
        )[0]
        self.gpt2_bos_token = tokenizers[1].convert_tokens_to_ids(
            [tokenizers[1].bos_token]
        )[0]
        self.gpt2_eos_token = tokenizers[1].convert_tokens_to_ids(
            [tokenizers[1].eos_token]
        )[0]

        global bert_pad_token
        global gpt2_pad_token
        bert_pad_token = self.bert_pad_token
        gpt2_pad_token = self.gpt2_pad_token

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "r") as handle:
                self.examples = json.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            dropped, count = self._read_corpus_natural_split(
                fname=file_path,
                max_length=block_size,
            )

            # random.shuffle(self.examples)
            logger.info("The number of dropped sentences is %d", dropped)
            logger.info("The number of processed sentences is %d", count)

            logger.info("Saving features into cached file %s", cached_features_file)

            with open(cached_features_file, "w") as handle:
                json.dump(self.examples, handle)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def collate(examples):
        input_ids_bert = pad_sequence(
            [torch.tensor(f["bert_token"], dtype=torch.long) for f in examples],
            batch_first=True,
            padding_value=bert_pad_token,
        )
        input_ids_gpt = pad_sequence(
            [torch.tensor(f["gpt2_token"], dtype=torch.long) for f in examples],
            batch_first=True,
            padding_value=gpt2_pad_token,
        )
        bert_token_length = torch.tensor(
            [f["bert_token_length"] for f in examples],
            dtype=torch.long,
        )
        gpt2_token_length = torch.tensor(
            [f["gpt2_token_length"] for f in examples],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [f["label"] for f in examples],
            dtype=torch.long,
        )

        return {
            "encoder_input_ids": input_ids_bert,
            "decoder_input_ids": input_ids_gpt,
            "encoder_input_lengths": bert_token_length,
            "decoder_input_lengths": gpt2_token_length,
            "pos_labels": labels,
        }

    def _read_corpus_natural_split(self, fname, max_length):
        dropped = 0
        count = 0
        import pandas as pd

        df = pd.read_csv(fname, header=0)
        for _, row in df.iterrows():
            label = row["label"]
            split_line_text = row["text"].strip()

            if len(split_line_text.split()) < 1:
                dropped += 1
                continue

            if max_length:
                if len(split_line_text.split()) >= max_length:
                    dropped += 1
                    continue

            if self.new_version:
                tokenized_text0 = self.tokenizers[0].encode(
                    split_line_text, max_length=max_length, truncation=True
                )
            else:
                tokenized_text0 = self.tokenizers[0].convert_tokens_to_ids(
                    self.tokenizers[0].tokenize(split_line_text)
                )
                tokenized_text0 = self.tokenizers[0].add_special_tokens_single_sentence(
                    tokenized_text0
                )
            tokenized_text0_length = len(tokenized_text0)

            tokenized_text1 = self.tokenizers[1].encode(
                split_line_text, max_length=max_length, truncation=True
            )
            tokenized_text1 = (
                [self.gpt2_bos_token] + tokenized_text1 + [self.gpt2_eos_token]
            )
            tokenized_text1_length = len(tokenized_text1)

            example = {
                "bert_token": tokenized_text0,
                "bert_token_length": tokenized_text0_length,
                "gpt2_token": tokenized_text1,
                "gpt2_token_length": tokenized_text1_length,
                "label": int(label),
            }

            self.examples.append(example)
            count += 1

        return dropped, count


class BucketBatchDataset(Dataset):
    def __init__(
        self,
        senti_file_path,
        topic_file_path,
        batch_size,
        tokenizers,
    ):
        self.examples = []

        self.senti_file_path = senti_file_path
        self.topic_file_path = topic_file_path

        assert any(
            [self.senti_file_path, self.topic_file_path]
        ), "eithor senti_file_path or topic_file_path should be non-empty!"

        if self.senti_file_path:
            self.senti_dataloader = BucketingDataLoader(
                senti_file_path,
                batch_size,
                tokenizers,
                bucket=100,
                shuffle=False,
            )

        if self.topic_file_path:
            self.topic_dataloader = BucketingDataLoader(
                topic_file_path,
                batch_size,
                tokenizers,
                bucket=100,
                shuffle=False,
            )

        self.load_from_dataloader()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def load_from_dataloader(self):
        if self.senti_file_path:
            for batch in self.senti_dataloader:
                batch["head_index"] = torch.tensor(0)
                self.examples.append(batch)

        if self.topic_file_path:
            for batch in self.topic_dataloader:
                batch["head_index"] = torch.tensor(1)
                self.examples.append(batch)

        random.shuffle(self.examples)

    @staticmethod
    def collate_fn(data):
        first = data[0]
        batch = {}
        for k in first.keys():
            batch[k] = data[0][k]
        return batch


class BucketSampler(Sampler):
    def __init__(
        self, lens, bucket_size, batch_size, droplast=False, shuffle=True, sample=False
    ):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle
        self.sample = sample

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [
            sorted(
                ids[i : i + self._bucket_size],
                key=lambda i: self._lens[i],
                reverse=True,
            )
            for i in range(0, len(ids), self._bucket_size)
        ]
        batches = [
            bucket[i : i + self._batch_size]
            for bucket in buckets
            for i in range(0, len(bucket), self._batch_size)
        ]
        if self._droplast:
            batches = [batch for batch in batches if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        if self.sample:
            batches = [
                ids[i * self._batch_size : (i + 1) * self._batch_size]
                for i in range(int(len(self._lens) / self._batch_size))
            ]
        return iter(batches)

    def __len__(self):
        bucket_sizes = [self._bucket_size] * (len(self._lens) // self._bucket_size) + [
            len(self._lens) % self._bucket_size
        ]
        if self._droplast:
            return sum(s // self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s / self._batch_size) for s in bucket_sizes)


class BucketingDataLoader:
    def __init__(
        self,
        file_path,
        batch_size,
        tokenizer,
        block_size=50,
        bucket=100,
        shuffle=True,
        sample=False,
    ):
        self.dataset = TokenDataset(tokenizer, file_path, block_size=block_size)
        self.batch_size = batch_size
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle
        self.num_examples = len(self.dataset.examples)
        self.num_batches = self.num_examples // batch_size
        self.example_lengths = [
            example["bert_token_length"] for example in self.dataset.examples
        ]
        self.sample = sample

    def __iter__(self):
        sampler = BucketSampler(
            self.example_lengths,
            self.bucket_size,
            self.batch_size,
            droplast=True,
            shuffle=self.shuffle,
            sample=self.sample,
        )
        loader = DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=TokenDataset.collate,
        )
        yield from loader

    def __len__(self):
        return self.num_batches

    def __del__(self):
        pass


if __name__ == "__main__":
    from transformers import AutoTokenizer

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    senti_file_path = ""
    topic_file_path = ""
    batch_size = 4
    tokenizers = [bert_tokenizer, gpt2_tokenizer]

    dataset = BucketBatchDataset(
        senti_file_path, topic_file_path, batch_size, tokenizers
    )
    sampler = torch.utils.data.RandomSampler(dataset)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=True,
        collate_fn=BucketBatchDataset.collate_fn,
        sampler=sampler,
    )
    print(next(iter(train_dataloader)))
