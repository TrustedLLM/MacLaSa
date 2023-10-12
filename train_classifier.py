# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn.init as init
from customized_transformers import *

from tqdm import tqdm, trange
from transformers import AutoTokenizer, AdamW

from transformers import get_polynomial_decay_schedule_with_warmup

from module import GAN
from module import VAE, DenseEmbedder
from utils import (
    BucketingDataLoader,
    sample_sequence_conditional,
)
from data_utils import build_dataload_and_cache_examples
import time

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": GPT2ForLatentConnector,
    "bert": BertForLatentConnector,
    "bertu": BertForLatentConnector,
}


from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset, DataLoader

model_id = "gpt2"
print(model_id)
model_ppl = GPT2LMHeadModel.from_pretrained(model_id).cuda()
tokenizer_ppl = GPT2TokenizerFast.from_pretrained(model_id)

start_time = time.time()


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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def weights_init_rondom(model):
    model = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        #         pdb.set_trace()
        if "encoder" in key:
            init.normal_(model_state_dict[key].data)
            # weight_init(item)


def save_cls_checkpoint(classifier, global_step, args, gan=None):
    save_last = args.save_step

    output_cls_dir = os.path.join(
        args.output_dir, "checkpoint-cls-{}".format(save_last)
    )
    if not os.path.exists(output_cls_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_cls_dir)

    if classifier is not None:
        logger.info("Saving classifier model checkpoint to %s", output_cls_dir)

        model_cls_to_save = (
            classifier.module if hasattr(classifier, "module") else classifier
        )  # Take care of distributed/parallel training

        checkpoint = {
            "iter": global_step,
            "model_state_dict": model_cls_to_save.state_dict(),
            "args": args,
        }
        torch.save(checkpoint, os.path.join(output_cls_dir, "training_cls.bin"))
        logger.info("Saving cls checkpoint to %s", output_cls_dir)
    if gan is not None:
        output_gan_dir = os.path.join(args.output_dir, "checkpoint-gan-{}".format("1"))
        if not os.path.exists(output_gan_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_gan_dir)
        logger.info("Saving GAN model checkpoint to %s", output_gan_dir)

        model_gan_to_save = (
            gan.module if hasattr(gan, "module") else gan
        )  # Take care of distributed/parallel training

        checkpoint_gan = {
            "iter": global_step,
            "model_state_dict": model_gan_to_save.state_dict(),
            "args": args,
        }
        torch.save(checkpoint_gan, os.path.join(output_gan_dir, "training_gan.bin"))
        logger.info("Saving GAN checkpoint to %s", output_gan_dir)


def access_latent_label(args, train_dataloader, model_vae, train=True):
    """Train the model"""
    # npy_file_path = (
    #     f"./data/train_{args.train_cls_gan}"
    #     if train
    #     else f"./data/eval_{args.train_cls_gan}"
    # )
    # if os.path.exists(npy_file_path + ".npy"):
    #     with open(npy_file_path + ".npy", "rb") as f:
    #         all_data = np.load(f)
    #         all_z = all_data[:, :-1]
    #         all_label = all_data[:, -1]
    # else:
    all_z = np.zeros((0, args.latent_size))
    all_label = np.zeros(
        (0),
    )
    epoch_iterator = tqdm(train_dataloader, desc="Creating Latent data")
    for step, batch in enumerate(epoch_iterator):
        batch["encoder_input_ids"] = batch["encoder_input_ids"].to(args.device)
        model_vae.eval()
        with torch.no_grad():
            latent_z = model_vae.encode_x(batch["encoder_input_ids"])
            all_z = np.append(all_z, latent_z.cpu().numpy(), 0)
            all_label = np.append(all_label, batch["pos_labels"].numpy(), 0)
    all_data = np.append(all_z, all_label[:, None], 1)
    # with open(npy_file_path + ".npy", "wb") as f:
    #     np.save(f, all_data)
    return [all_z, all_label]


def train(
    args,
    train_dataloader,
    model_vae,
    decoder_tokenizer,
    gan,
    eval_latent_dataset,
):
    """Train the model"""

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
    # Prepare optimizer and schedule (linear warmup and decay)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in gan.latent_discriminator.named_parameters()],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    optimizer_grouped_parameters_G = [
        {
            "params": [p for n, p in gan.latent_generator.named_parameters()],
            "weight_decay": 0.0,
        },
    ]

    optimizer_G = AdamW(
        optimizer_grouped_parameters_G, lr=args.learning_rate, eps=args.adam_epsilon
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

        gan.latent_discriminator, optimizer = amp.initialize(
            gan.latent_discriminator, optimizer, opt_level=args.fp16_opt_level
        )
        gan.latent_generator, optimizer_G = amp.initialize(
            gan.latent_generator, optimizer_G, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_vae = torch.nn.DataParallel(model_vae, device_ids=range(args.n_gpu)).to(
            args.device
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    train_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model_vae.zero_grad()

    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    n_iter = int(args.num_train_epochs) * len(train_dataloader)

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    args.logging_steps = int(np.floor(len(train_dataloader)))
    args.save_steps = args.logging_steps
    best_gan_diff = 20000
    best_acc_cnt = 0
    best_diff_cnt = 0
    loss_gan_g = torch.tensor(0)
    gan_d_weight = 1
    stop_flag = False
    start_time = time.time()
    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        if best_gan_diff < 20000:
            use_time = time.time() - start_time
            start_time = time.time()
            logger.info("Time for this epoch = %f", use_time)
        for step, batch in enumerate(epoch_iterator):
            latent_z = batch["latent_z"].float().to(args.device)

            model_vae.eval()
            gan.train()

            loss_gan_d = gan.d_loss(latent_z)

            loss = gan_d_weight * loss_gan_d
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(gan.parameters(), args.max_grad_norm)

                optimizer.step()

                gan.latent_discriminator.zero_grad()

                if step % args.n_cyc == 0:
                    loss_gan_g = gan.g_loss(latent_z)
                    loss_gan_g.backward()
                    optimizer_G.step()
                    gan.latent_generator.zero_grad()
                epoch_iterator.set_description(
                    (
                        f"iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; "
                        f"loss_d: {loss_gan_d.item():.3f}; loss_g: {loss_gan_g.item():.3f}; "
                    )
                )
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate_acc(
                            args,
                            classifier=None,
                            gan=gan,
                            eval_latent_dataset=eval_latent_dataset,
                        )
                        results.update(
                            calc_ppl_lgy(
                                model_vae,
                                decoder_tokenizer,
                                args,
                                gan,
                                model_ppl,
                                tokenizer_ppl,
                            )
                        )

                        logger.info("GAN Dis ACC = %f", results["gan_acc_diff"])
                        logger.info("PPL = %f", results["ppl"])
                        logger.info("sBLEU = %f", results["sbleu"])
                        logger.info("PPL+sBLEU = %f", results["ppl+sbleu"])
                        logger.info("Length = %f", results["length"])
                        logger.info(
                            "z norm = %f--%f", results["norm_z"], results["true_norm_z"]
                        )

                        if results["ppl+sbleu"] < best_gan_diff:
                            best_gan_diff = results["ppl+sbleu"]
                            best_diff_cnt = 0
                            save_cls_checkpoint(None, global_step, args, gan=gan)
                        else:
                            best_diff_cnt += 1

            if (best_acc_cnt >= 3 and best_diff_cnt > 10) or stop_flag:
                break
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return 0


def train_cls(
    args,
    train_dataloader,
    model_vae,
    classifier,
    gan,
    latent_dataset,
    eval_latent_dataset,
):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
    # Prepare optimizer and schedule (linear warmup and decay)

    optimizer_grouped_parameters = [
        {"params": [p for n, p in classifier.named_parameters()], "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, args.warmup_steps, num_training_steps=t_total, lr_end=5e-7, power=3.0
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        classifier, optimizer = amp.initialize(
            classifier, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_vae = torch.nn.DataParallel(model_vae, device_ids=range(args.n_gpu)).to(
            args.device
        )

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_vae = torch.nn.parallel.DistributedDataParallel(
            model_vae,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    global_step = 0
    train_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model_vae.zero_grad()

    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    args.logging_steps = int(np.floor(len(train_dataloader)))
    args.save_steps = args.logging_steps
    best_cls_acc = -10
    best_acc_cnt = 0

    best_cls_train_acc = -10
    stop_flag = False
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

        for step, batch in enumerate(epoch_iterator):
            latent_z = batch["latent_z"].float().to(args.device)
            latent_labels = batch["labels"].long().to(args.device)

            model_vae.eval()
            classifier.train()
            logits = classifier(latent_z)
            num_classes = logits.size(-1)
            if num_classes > 1:
                loss = torch.nn.CrossEntropyLoss()(logits, latent_labels)
            else:
                loss = torch.norm(logits - latent_labels[:, None], dim=1) ** 2 * 0.5
            # loss_rec, loss_kl, loss, mu, std = model_vae(inputs, labels,std=True)
            train_step += 1
            loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        classifier.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                # scheduler_D.step()
                classifier.zero_grad()

                epoch_iterator.set_description(
                    (
                        f"iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; "
                    )
                )
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate_acc(
                            args,
                            classifier=classifier,
                            gan=gan,
                            eval_latent_dataset=eval_latent_dataset,
                        )
                        results.update(
                            evaluate_train_acc(
                                args,
                                classifier=classifier,
                                gan=gan,
                                latent_dataset=latent_dataset,
                            )
                        )
                        train_iterator.set_description(
                            "Train ACC:"
                            + str(round(results["train-acc"], 2)).ljust(5)
                            + " Test ACC:"
                            + str(round(results["acc"], 2)).ljust(5)
                        )

                        if results["acc"] > best_cls_acc:
                            if results["train-acc"] > best_cls_train_acc:
                                best_cls_train_acc = results["train-acc"]
                            best_cls_acc = results["acc"]
                            best_acc_cnt = 0

                            save_cls_checkpoint(
                                classifier,
                                global_step,
                                args,
                                gan=None,
                            )
                        elif (
                            results["acc"] == best_cls_acc
                            and results["train-acc"] > best_cls_train_acc
                        ):
                            best_cls_train_acc = results["train-acc"]
                            best_acc_cnt = 0
                            save_cls_checkpoint(
                                classifier,
                                global_step,
                                args,
                                gan=None,
                            )
                        else:
                            best_acc_cnt += 1

            if (best_acc_cnt >= 5) or stop_flag:
                break
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return 0


def calc_ppl_lgy(
    model_vae,
    decoder_tokenizer,
    args,
    gan=None,
    model=None,
    tokenizer=None,
):
    generate_text = []
    bz = 50
    num_epoch = 250 // bz

    for batch in trange(num_epoch, desc="Evaluating PPL", disable=True):
        latent_z = gan.generate_z(bz, eval=True)
        context_tokens = decoder_tokenizer.encode("<BOS>")
        with torch.no_grad():
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=latent_z,
                length=50,
                num_samples=latent_z.size(0),
                device=args.device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=model_vae.eos_token_id,
            )
        for i in range(latent_z.size(0)):
            text_x1 = (
                decoder_tokenizer.decode(
                    out[i, :].tolist(), clean_up_tokenization_spaces=False
                )
                .split("<EOS>")[0]
                .replace("<BOS>", "")
                .strip()
            )
            text_x1 = " ".join(text_x1.split())
            generate_text.append(text_x1 + "\n")
    encodings = tokenizer("\n\n".join(generate_text), return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    list_of_references = []
    len_list = []
    for jj, line in enumerate(generate_text):
        if jj < 10:
            print(line)
        split = line.strip().split(" ")
        list_of_references.append(split)
        len_list.append(len(split))
    sbleu = []
    num_all = len(list_of_references)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    for i in range(num_all):
        refs = [list_of_references[j] for j in range(num_all) if i != j]
        bleu_ = sentence_bleu(
            refs, list_of_references[i], smoothing_function=SmoothingFunction().method1
        )
        sbleu.append(bleu_ * 100)
    score = np.mean(sbleu)
    len_mean = np.mean(len_list)
    norm_z = latent_z.norm(dim=-1).mean().item()
    return {
        "ppl": ppl,
        "sbleu": round(score, 2),
        "length": round(len_mean, 2),
        "norm_z": norm_z,
        "ppl+sbleu": ppl + round(score, 2),
    }


def evaluate_acc(
    args,
    classifier=None,
    gan=None,
    eval_latent_dataset=None,
):
    eval_dataloader = DataLoader(
        eval_latent_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    # eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=True)
    corrects = []
    acc_diff_list = []
    neg_corrects = []
    pos_corrects = []
    neg_cnt = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating acc", disable=True):
        latent_z = batch["latent_z"].float().to(args.device)
        latent_labels = batch["labels"].long().to(args.device)
        with torch.no_grad():
            if classifier:
                logits = classifier(latent_z)
                loss = torch.nn.CrossEntropyLoss()(logits, latent_labels)
                num_classes = logits.size(-1)
                if num_classes > 1:
                    neg_cnt += (latent_labels == 0).sum()
                    correct = logits.max(1)[1] == latent_labels.long()
                    neg_correct = (latent_labels.long() == 0) & (logits.max(1)[1] == 0)
                    pos_correct = (latent_labels.long() == 1) & (logits.max(1)[1] == 1)

                else:
                    latent_labels = latent_labels.float()
                    correct = (
                        -torch.norm(logits - latent_labels[:, None], dim=1) ** 2 * 0.5
                    )
                corrects.extend(correct.float().cpu().numpy())
                neg_corrects.extend(neg_correct.float().cpu().numpy())
                pos_corrects.extend(pos_correct.float().cpu().numpy())
                loss_out = round(loss.mean().item(), 3)
            else:
                correct = loss_out = neg_correct = pos_correct = 0
            ### GAN
            gan_acc = gan.discriminate_acc(latent_z)
            acc_diff_list.append(gan_acc)
    norm_z = latent_z.norm(dim=-1).mean().item()
    correct = np.mean(corrects)
    neg_correct = np.sum(neg_corrects) / neg_cnt
    pos_correct = np.sum(pos_corrects) / (len(eval_latent_dataset) - neg_cnt)
    gan_acc_diff = np.mean(acc_diff_list)

    return {
        "acc": correct,
        "gan_acc_diff": gan_acc_diff,
        "loss": loss_out,
        "neg_acc": neg_correct,
        "pos_acc": pos_correct,
        "true_norm_z": norm_z,
    }


def evaluate_train_acc(
    args,
    classifier=None,
    gan=None,
    latent_dataset=None,
):
    eval_dataloader = DataLoader(
        latent_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    # eval_dataloader = build_dataload_and_cache_examples(args, [encoder_tokenizer, decoder_tokenizer], evaluate=False)
    corrects = []
    acc_diff_list = []
    i = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating train acc", disable=True):
        latent_labels = batch["labels"].long().to(args.device)
        latent_z = batch["latent_z"].float().to(args.device)
        with torch.no_grad():
            logits = classifier(latent_z)
            loss = torch.nn.CrossEntropyLoss()(logits, latent_labels)
            num_classes = logits.size(-1)
            if num_classes > 1:
                correct = logits.max(1)[1] == latent_labels.long()
            else:
                latent_labels = latent_labels.float()
                correct = -torch.norm(logits - latent_labels[:, None], dim=1) ** 2 * 0.5
            corrects.extend(correct.float().cpu().numpy())
            ### GAN
            gan_acc = gan.discriminate_acc(latent_z)
            acc_diff_list.append(gan_acc)

    correct = np.mean(corrects)
    gan_acc_diff = np.mean(acc_diff_list)
    return {
        "train-acc": correct,
        "train-gan_acc_diff": gan_acc_diff,
        "train-loss": round(loss.mean().item(), 3),
    }


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--train_senti_data_file",
        default="",
        type=str,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--train_topic_data_file",
        default="",
        type=str,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="",
        type=str,
        help="The directory where checkpoints are saved.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--eval_senti_data_file",
        default="",
        # default="",
        type=str,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--eval_topic_data_file",
        default="",
        # default="",
        type=str,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--save_bert_gpt_init", action="store_true", help="Use Philly for computing."
    )
    parser.add_argument(
        "--length_weighted_loss",
        action="store_true",
        help="Use sentence length re-weight the reconstruction loss.",
    )

    ## Encoder options
    parser.add_argument(
        "--encoder_model_type",
        default="bert",
        type=str,
        help="The encoder model architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--encoder_model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="The encoder model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--encoder_config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--encoder_tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )

    ## Decoder options
    parser.add_argument(
        "--decoder_model_type",
        default="gpt2",
        type=str,
        help="The decoder model architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--decoder_model_name_or_path",
        default="gpt2-medium",
        type=str,
        help="The decoder model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--decoder_config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--decoder_tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )

    ## Variational auto-encoder
    parser.add_argument(
        "--latent_size", default=128, type=int, help="Latent space dimension."
    )
    parser.add_argument(
        "--use_deterministic_connect",
        action="store_true",
        help="Use deterministic inference to generate latent codes, i.e., standard auto-encoders.",
    )
    parser.add_argument(
        "--use_pretrained_model",
        action="store_true",
        help="Use pre-trained auto-encoder models as the initialization",
    )
    parser.add_argument(
        "--latent_as_gpt_memory",
        default=1,
        type=int,
        help="Latent vector as memery for GPT2 to attend.",
    )
    parser.add_argument(
        "--latent_as_gpt_emb",
        default=1,
        type=int,
        help="Latent vector as embeddings for GPT2.",
    )

    ## Objective functions
    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="The weighting hyper-parameter of the KL term in VAE",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="Optional input sequence length before tokenization. The sequence will be dropped if it is longer the max_seq_length",
    )
    parser.add_argument(
        "--block_size",
        default=50,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_eval_rec",
        action="store_true",
        help="Whether to run eval reconstruction on a set of models.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    # Training Schedules
    parser.add_argument(
        "--ratio_increase",
        default=0.25,
        type=float,
        help="Learning schedule, the percentage for the annealing stage.",
    )
    parser.add_argument(
        "--ratio_zero",
        default=0.5,
        type=float,
        help="Learning schedule, the percentage for the pure auto-encoding stage.",
    )
    parser.add_argument(
        "--fb_mode", default=0, type=int, help="free bit training mode."
    )
    parser.add_argument(
        "--dim_target_kl",
        default=0.9,
        type=float,
        help="dim_target_kl free bit training mode.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=64,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=5,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--use_pretrained_vae",
        action="store_true",
        help="Use use_pretrained_vae as initialization, where beta value is specified in the folder",
    )
    parser.add_argument(
        "--use_random_weight",
        action="store_true",
        help="Use random weights as initialization",
    )

    ## IO: Logging and Saving
    parser.add_argument(
        "--logging_steps", type=float, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gloabl_step_eval",
        type=int,
        default=1,
        help="Evaluate the results at the given global step",
    )

    # Precision & Distributed Training
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--train_cls_gan", type=str, default="cls")
    parser.add_argument("--n_cyc", type=int, default=5)
    parser.add_argument("--save_step", type=str, default=2)
    parser.add_argument(
        "--fix_model",
        type=int,
        default=84,
        help="0: no fix; 1: fix both bert & gpt; 2: fix gpt; 3: fix both bert & gpt, extra layers",
    )
    args = parser.parse_args()

    if args.train_cls_gan == "cls":
        if args.save_step == 1:
            args.train_topic_data_file = ""
            args.eval_topic_data_file = ""
        elif args.save_step == 2:
            args.train_senti_data_file = ""
            args.eval_senti_data_file = ""

    args.do_train = True
    args.do_eval = True
    args.overwrite_output_dir = True
    args.evaluate_during_training = True
    args.use_pretrained_model = True
    args.do_lower_case = True if "uncased" in args.encoder_model_name_or_path else False
    args.output_dir = args.checkpoint_dir

    if args.fix_model == 84:
        MODEL_CLASSES["bertu"] = BertForLatentConnectorAVG
        if "large" in args.decoder_model_name_or_path:
            MODEL_CLASSES["gpt2"] = GPT2ForLatentConnectorNew
        else:
            MODEL_CLASSES["gpt2"] = GPT2ForLatentConnectorNew2

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    global_step = args.gloabl_step_eval

    output_full_dir = os.path.join(
        args.checkpoint_dir, "checkpoint-full-{}".format(global_step)
    )

    checkpoint = torch.load(os.path.join(output_full_dir, "training.bin"))

    ## Encoder
    encoder_model_class = MODEL_CLASSES[args.encoder_model_type]
    tokenizer_encoder = AutoTokenizer.from_pretrained(
        args.encoder_tokenizer_name
        if args.encoder_tokenizer_name
        else args.encoder_model_name_or_path,
        do_lower_case=args.do_lower_case,
    )

    if args.block_size <= 0:
        args.block_size = (
            tokenizer_encoder.max_len_single_sentence
        )  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_encoder.max_len_single_sentence)

    model_encoder = encoder_model_class.from_pretrained(
        args.encoder_model_name_or_path,
        latent_size=args.latent_size,
        pad_id=tokenizer_encoder.pad_token_id,
    )

    ## Decoder
    decoder_model_class = MODEL_CLASSES[args.decoder_model_type]
    tokenizer_decoder = AutoTokenizer.from_pretrained(
        args.decoder_tokenizer_name
        if args.decoder_tokenizer_name
        else args.decoder_model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    if args.block_size <= 0:
        args.block_size = (
            tokenizer_decoder.max_len_single_sentence
        )  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer_decoder.max_len_single_sentence)

    if args.latent_as_gpt_emb + args.latent_as_gpt_memory == 0:
        return  # latent vector should pass into GPT to decode
    else:
        latent_as_gpt_emb = True if args.latent_as_gpt_emb == 1 else False
        latent_as_gpt_memory = True if args.latent_as_gpt_memory == 1 else False

    model_decoder = decoder_model_class.from_pretrained(
        args.decoder_model_name_or_path,
        latent_size=args.latent_size,
        latent_as_gpt_emb=latent_as_gpt_emb,
        latent_as_gpt_memory=latent_as_gpt_memory,
    )

    if args.fix_model == 84:
        print("Initialize the Extra Layer.")
        model_decoder.transformer.change_order()

    # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
    }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "tokens to GPT2")
    model_decoder.resize_token_embeddings(
        len(tokenizer_decoder)
    )  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == "<PAD>"

    model_vae = VAE(
        model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args
    )

    if args.use_pretrained_model:
        model_vae.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device)  #
    gan = GAN(args)
    gan.to(args.device)
    gan.apply(weights_init_rondom)

    classifier = DenseEmbedder(args.latent_size, 2, depth=4, num_classes=args.n_classes)
    classifier.to(args.device)
    classifier.apply(weights_init_rondom)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    ##############################
    # Training
    global_step = 0
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

    train_dataloader = build_dataload_and_cache_examples(
        args, [tokenizer_encoder, tokenizer_decoder], evaluate=False
    )

    all_z, all_label = access_latent_label(
        args, train_dataloader, model_vae, train=True
    )
    latent_dataset = LatentDataset(all_z, all_label)

    dataloader = DataLoader(
        latent_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    eval_dataloader = build_dataload_and_cache_examples(
        args, [tokenizer_encoder, tokenizer_decoder], evaluate=True
    )
    eval_z, eval_label = access_latent_label(
        args, eval_dataloader, model_vae, train=False
    )
    eval_latent_dataset = LatentDataset(eval_z, eval_label)
    if "gan" in args.train_cls_gan:
        train(
            args,
            dataloader,
            model_vae,
            tokenizer_decoder,
            gan=gan,
            eval_latent_dataset=eval_latent_dataset,
        )
    else:
        print("Load ")
        gan_dir = os.path.join(args.output_dir, "checkpoint-gan-1/training_gan.bin")
        gan_checkpoint = torch.load(gan_dir)
        gan.load_state_dict(gan_checkpoint["model_state_dict"], strict=True)
    if "cls" in args.train_cls_gan:
        global start_time
        start_time = time.time()
        latent_aug_dataset = LatentDataset(all_z, all_label)
        dataloader = DataLoader(
            latent_aug_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        train_cls(
            args,
            dataloader,
            model_vae,
            classifier=classifier,
            gan=gan,
            latent_dataset=latent_dataset,
            eval_latent_dataset=eval_latent_dataset,
        )


if __name__ == "__main__":
    main()
