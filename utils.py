import os
import json
import math
import logging
import random

from tqdm import tqdm

import torch
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


def sample_sequence_conditional(
    model,
    length,
    context,
    past=None,
    num_samples=1,
    temperature=1,
    device="cpu",
    decoder_tokenizer=None,
    eos_id=50259,
    loss=False,
):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context

    with torch.no_grad():
        for ii in range(length):
            inputs = {"input_ids": generated, "past": past}
            outputs = model(
                **inputs
            )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / temperature
            # next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            next_token = F.softmax(next_token_logits, dim=-1).max(-1, keepdim=True)[1]
            generated = torch.cat((generated, next_token), dim=1)
            tmp = next_token.squeeze() == eos_id
            if ii == 0:
                tmp22 = torch.zeros_like(tmp, device=device)
            tmp22 = torch.logical_or(tmp22, tmp)
            if False not in tmp22:
                break
        if loss:
            outputs = model(
                input_ids=generated,
                past=past,
                labels=generated,
                label_ignore=decoder_tokenizer.pad_token_id,
            )
            rec_loss = (-outputs[0]).tolist()
            return generated, rec_loss
    return generated


class LatentDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, latent_z, labels):
        self.latent_z = latent_z
        self.labels = labels

    def __len__(self):
        return len(self.latent_z)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {"latent_z": self.latent_z[idx], "labels": self.labels[idx]}
        return sample


class TokenDataset(Dataset):
    def __init__(
        self,
        tokenizers,
        args,
        file_path="train",
        block_size=512,
    ):
        print("file:\t", file_path)
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_lm_"
            + args.decoder_model_type
            + "_"
            + args.encoder_model_type
            + f"_{block_size}_{filename[:-4]}.json",
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
        # Convert to Tensors and build dataset

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
        try:
            token_lengths = torch.tensor(
                [[f["bert_token_length"], f["gpt2_token_length"]] for f in examples],
                dtype=torch.long,
            )
        except:
            token_lengths = torch.zeros((len(examples), 1091))
            for i in range(len(examples)):
                token_lengths[i, examples[i]["gpt2_token_length"]] = 1
        return (input_ids_bert, input_ids_gpt, token_lengths)

    def _read_corpus_natural_split(self, fname, max_length):
        dropped = 0
        count = 0
        with open(fname) as fin:
            for line in fin:
                split_line = line.split("\t")
                lb = split_line[0]
                split_line_text = split_line[1]

                if len(split_line_text.split()) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line_text.split()) > max_length:
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
                    tokenized_text0 = self.tokenizers[
                        0
                    ].add_special_tokens_single_sentence(tokenized_text0)

                tokenized_text0_length = len(tokenized_text0)

                tokenized_text1 = self.tokenizers[1].encode(
                    " " + split_line_text, max_length=max_length, truncation=True
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
                }

                example["gpt2_token_length"] = float(lb)

                self.examples.append(example)
                count += 1

        return dropped, count


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
        max_seq_length,
        tokenizer,
        args,
        bucket=100,
        shuffle=True,
        sample=False,
    ):
        self.dataset = TokenDataset(
            tokenizer, args, file_path, block_size=args.block_size
        )
        self.batch_size = batch_size
        self.max_len = max_seq_length
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


def weights_init_rondom(model):
    model = model.module if hasattr(model, "module") else model
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        if "encoder" in key:
            init.normal_(model_state_dict[key].data)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def frange_cycle_zero_linear(
    n_iter, start=0.0, stop=1.0, n_cycle=4, ratio_increase=0.5, ratio_zero=0.3
):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio_increase)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            if i < period * ratio_zero:
                L[int(i + c * period)] = start
            else:
                L[int(i + c * period)] = v
                v += step
            i += 1
    return L


def build_dataload_and_cache_examples(args, tokenizer, evaluate=False):
    if isinstance(tokenizer, list):
        if not evaluate:
            args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            file_path = args.train_data_file
        else:
            args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            file_path = args.eval_data_file
        dataloader = BucketingDataLoader(
            file_path,
            args.batch_size,
            args.max_seq_length,
            tokenizer,
            args,
            bucket=100,
            shuffle=False,
        )
    else:
        pass
    return dataloader


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def calc_iwnll(model_vae, eval_dataloader, args):
    report_kl_loss = report_rec_loss = report_loss = 0
    report_num_words = report_num_sents = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating PPL", disable=args.disable_bar):

        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(args.device)

        x0_max_lengths, _ = batch["encoder_input_lengths"].max(dim=0)
        x1_max_lengths, _ = batch["decoder_input_lengths"].max(dim=0)

        batch["encoder_input_ids"] = batch["encoder_input_ids"][:, :x0_max_lengths]
        batch["decoder_input_ids"] = batch["decoder_input_ids"][:, :x1_max_lengths]

        report_num_words += batch["decoder_input_lengths"].sum().item()
        report_num_sents += args.eval_batch_size
        with torch.no_grad():
            loss, loss_rc, loss_kl = model_vae.loss_iw(
                batch["encoder_input_ids"], batch["decoder_input_ids"], nsamples=1
            )

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss = loss.sum()

        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()
        report_loss += loss.item()

    elbo = (report_kl_loss - report_rec_loss) / report_num_sents
    nll = -report_rec_loss / report_num_sents  # mean of rec loss
    kl = report_kl_loss / report_num_sents
    ppl = np.exp(-report_loss / report_num_words)

    return ppl, elbo, nll, kl
