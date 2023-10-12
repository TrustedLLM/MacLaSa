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
from customized_transformers import *
import argparse
import logging
import os
import time
import json
import random

import numpy as np
import torch
import torch.nn.init as init

from nltk.translate.bleu_score import corpus_bleu
from transformers import (
    AdamW,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import AutoTokenizer

from module import VAE
from data_utils import build_dataload_and_cache_examples
from utils import (
    calc_iwnll,
    frange_cycle_zero_linear,
)


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": GPT2ForLatentConnector,
    "bertu": BertForLatentConnector,
    "bert": BertForLatentConnector,
}

parameter_name = []


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
        if "encoder" in key:
            init.normal_(model_state_dict[key].data)


def save_checkpoint(model_vae, global_step, args):
    save_last = 1
    model_to_save = (
        model_vae.module if hasattr(model_vae, "module") else model_vae
    )  # Take care of distributed/parallel training
    state_dict_new = {}
    state_dict = model_to_save.state_dict()
    for key in parameter_name:
        state_dict_new[key] = state_dict[key]
    checkpoint = {
        "iter": global_step,
        "model_state_dict": state_dict_new,
        "beta": model_to_save.args.beta,
        "args": args,
    }

    output_full_dir = os.path.join(
        args.output_dir, "checkpoint-full-{}".format(save_last)
    )
    if not os.path.exists(output_full_dir) and args.local_rank in [-1, 0]:
        os.makedirs(output_full_dir)

    logger.info("Start saving full model checkpoint to %s", output_full_dir)
    torch.save(checkpoint, os.path.join(output_full_dir, "training.bin"))
    logger.info("Saving full checkpoint to %s", output_full_dir)


def train(
    args,
    train_dataloader,
    model_vae,
    encoder_tokenizer,
    decoder_tokenizer,
):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(
            "./runs/"
            + args.output_dir.split("/")[-2]
            + "/"
            + args.output_dir.split("/")[-1]
        )

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
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model_vae.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model_vae.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    parameter_name.extend([n for n, p in model_vae.named_parameters()])

    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )

    from transformers import (
        get_polynomial_decay_schedule_with_warmup,
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
        model_vae, optimizer = amp.initialize(
            model_vae, optimizer, opt_level=args.fp16_opt_level
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

    model_vae = (
        model_vae.module if hasattr(model_vae, "module") else model_vae
    )  # Take care of distributed/parallel training

    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.disable_bar
    )

    n_iter = int(args.num_train_epochs) * len(train_dataloader)
    beta_t_list = frange_cycle_zero_linear(
        n_iter,
        start=0.0,
        stop=args.beta,
        n_cycle=int(args.num_train_epochs),
        ratio_increase=args.ratio_increase,
        ratio_zero=args.ratio_zero,
    )

    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    with torch.no_grad():
        result_new = calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args)
        result_new.update(
            evaluate(args, model_vae, encoder_tokenizer, decoder_tokenizer)
        )
        for key, value in result_new.items():
            logger.info("eval_%s:%f", key, value)
            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

    logger.info('\nBLEU is %f\n"', result_new["bleu"])

    args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps
    args.save_steps = args.logging_steps
    best_bleu = 0
    final_beta = args.beta

    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.disable_bar
        )
        for step, batch in enumerate(epoch_iterator):

            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(args.device)

            model_vae.train()
            beta_t = beta_t_list[step + epoch * len(epoch_iterator)]
            # model_vae.args.fb_mode = 1
            if args.n_gpu == 1:
                model_vae.args.beta = beta_t
                if beta_t == 0.0:
                    model_vae.args.fb_mode = 3
                else:
                    model_vae.args.fb_mode = 1

                if args.use_deterministic_connect:
                    model_vae.args.fb_mode = 2
                if final_beta == 0.0:
                    model_vae.args.fb_mode = 3
            else:
                model_vae.module.args.beta = beta_t

                if beta_t == 0.0:
                    model_vae.module.args.fb_mode = 0
                else:
                    model_vae.module.args.fb_mode = 1

                if args.use_deterministic_connect:
                    model_vae.module.args.fb_mode = 2

            (
                loss_rec,
                loss_kl,
                latent_classify_loss,
                gap_loss,
                loss,
                mu,
                std,
            ) = model_vae(
                encoder_input_ids=batch["encoder_input_ids"],
                decoder_input_ids=batch["decoder_input_ids"],
                pos_labels=batch["pos_labels"],
                head_index=batch["head_index"],
                std=True,
            )

            if train_step % 100 == 0:
                tb_writer.add_scalar(
                    "loss_rec_train", loss_rec.mean().item(), train_step
                )
                tb_writer.add_scalar("loss_kl_train", loss_kl.mean().item(), train_step)
                tb_writer.add_scalar(
                    "classify_loss_train",
                    latent_classify_loss.mean().item(),
                    train_step,
                )
                if gap_loss:
                    tb_writer.add_scalar(
                        "gap_loss_train", gap_loss.mean().item(), train_step
                    )
                tb_writer.add_scalar("loss_train", loss.mean().item(), train_step)
                tb_writer.add_scalar("beta_train", beta_t, train_step)
                tb_writer.add_scalar("lr_train", scheduler.get_last_lr()[0], train_step)
                tb_writer.add_scalar("std", std.mean().item(), train_step)
                tb_writer.add_scalar("mean", mu.mean().item(), train_step)

            train_step += 1

            loss_rec = (
                loss_rec.mean()
            )  # mean() to average on multi-gpu parallel training
            loss_kl = loss_kl.mean()
            latent_classify_loss = latent_classify_loss.mean()
            if gap_loss:
                gap_loss = gap_loss.mean()
            loss = loss.mean()

            if args.n_gpu == 1:
                beta_ = model_vae.args.beta
            else:
                beta_ = model_vae.module.args.beta

            if train_step % (args.logging_steps // 10) == 0:
                epoch_iterator.set_description(
                    (
                        f"iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; "
                        f"loss_rec: {loss_rec.item():.3f}; loss_kl: {loss_kl.item():.3f}; classify_loss: {latent_classify_loss.item():.3f};  gap_loss: {gap_loss.item():.3f}; "
                        f"beta: {beta_:.3f}"
                    )
                )

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
                        model_vae.parameters(), args.max_grad_norm
                    )

                optimizer.step()

                scheduler.step()

                model_vae.zero_grad()

                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(
                            args,
                            model_vae,
                            encoder_tokenizer,
                            decoder_tokenizer,
                        )
                        results.update(
                            calc_rec_lgy(
                                model_vae,
                                encoder_tokenizer,
                                decoder_tokenizer,
                                args,
                            )
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                    if results["bleu"] >= best_bleu:
                        best_bleu = results["bleu"]
                        if not args.no_save:
                            save_checkpoint(model_vae, global_step, args)

                            # save best bleu to file
                            output_eval_file = os.path.join(
                                args.output_dir, "eval_results_bleu.txt"
                            )
                            if not os.path.exists(args.output_dir):
                                os.makedirs(args.output_dir)
                            with open(output_eval_file, "w") as writer:
                                writer.write("%s = %s\n" % ("bleu", str(best_bleu)))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    results = calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args)
    for key, value in results.items():
        tb_writer.add_scalar("final_{}".format(key), value, global_step)
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, optimizer


def evaluate(
    args,
    model_vae,
    encoder_tokenizer,
    decoder_tokenizer,
    prefix="",
    subset="test",
):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    logger.info("***** Running evaluation on {} dataset *****".format(subset))

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_dataloader = build_dataload_and_cache_examples(
        args, [encoder_tokenizer, decoder_tokenizer], evaluate=True
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model_vae.eval()

    model_vae = (
        model_vae.module if hasattr(model_vae, "module") else model_vae
    )  # Take care of distributed/parallel training

    ppl, elbo, nll, kl = calc_iwnll(model_vae, eval_dataloader, args)

    result = {"perplexity": ppl, "elbo": elbo, "kl": kl, "nll": nll}

    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def calc_rec_lgy(model_vae, encoder_tokenizer, decoder_tokenizer, args):
    from utils import sample_sequence_conditional

    eval_dataloader = build_dataload_and_cache_examples(
        args, [encoder_tokenizer, decoder_tokenizer], evaluate=True
    )
    count = 0

    ref = []
    cand = []

    for batch in tqdm(
        eval_dataloader, desc="Evaluating recontruction", disable=args.disable_bar
    ):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(args.device)

        x0 = batch["encoder_input_ids"]
        x1 = batch["decoder_input_ids"]

        x0_max_lengths, _ = batch["encoder_input_lengths"].max(dim=0)
        x1_max_lengths, _ = batch["decoder_input_lengths"].max(dim=0)

        x0 = x0[:, :x0_max_lengths]
        x1 = x1[:, :x1_max_lengths]

        context_tokens = decoder_tokenizer.encode("<BOS>")
        with torch.no_grad():
            attention_mask = (x0 != encoder_tokenizer.pad_token_id).float()

            pooled_hidden_fea = model_vae.encoder(x0, attention_mask)[1]

            mean, logvar = model_vae.encoder.linear(pooled_hidden_fea).chunk(2, -1)
            latent_z = mean.squeeze(1)

            past = latent_z
            out = sample_sequence_conditional(
                model=model_vae.decoder,
                context=context_tokens,
                past=past,
                length=x1_max_lengths,
                num_samples=latent_z.size(0),
                device=args.device,
                decoder_tokenizer=decoder_tokenizer,
                eos_id=model_vae.eos_token_id,
            )

        for i in range(latent_z.size(0)):
            text_x0_ = (
                decoder_tokenizer.decode(
                    x1[i, :].tolist(), clean_up_tokenization_spaces=False
                )
                .split("<EOS>")[0]
                .replace("<BOS>", "")
                .strip()
            )
            text_x0_ = text_x0_.split()
            text_x1 = (
                decoder_tokenizer.decode(
                    out[i, :].tolist(), clean_up_tokenization_spaces=False
                )
                .split("<EOS>")[0]
                .replace("<BOS>", "")
                .strip()
            )
            text_x1 = text_x1.split()

            count += 1
            ref.append([text_x0_])
            cand.append(text_x1)

        if count > 1000:
            break
    bleu = corpus_bleu(ref, cand) * 100
    logger.info("  BLEU = %s", str(round(bleu, 2)))
    return {"bleu": bleu}


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--train_senti_data_file",
        default="",
        # default="",
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
    parser.add_argument("--latent_loss_weight", type=float, default=1, help="")
    parser.add_argument("--gap_loss_weight", type=float, default=1, help="")
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
        default=0.9,
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
        "--fb_mode", default=1, type=int, help="free bit training mode."
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
        default=1e-4,
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
        default=50,
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
        "--use_philly", action="store_true", help="Use Philly for computing."
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
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
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
        "--seed", type=int, default=123, help="random seed for initialization"
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
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument("--fix_model", type=int, default=84, help="fix model.")

    parser.add_argument("--disable_bar", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    if args.fix_model == 84:
        MODEL_CLASSES["bertu"] = BertForLatentConnectorAVG
        if "large" in args.decoder_model_name_or_path:
            MODEL_CLASSES["gpt2"] = GPT2ForLatentConnectorNew
        else:
            MODEL_CLASSES["gpt2"] = GPT2ForLatentConnectorNew2

    args.do_train = True
    args.do_eval = True
    args.overwrite_output_dir = True
    args.length_weighted_loss = True
    args.evaluate_during_training = True
    args.do_lower_case = True if "uncased" in args.encoder_model_name_or_path else False
    # args.fp16 = True

    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    output_name = (
        time_str
        + "_seed"
        + str(args.seed)
        + "_"
        + args.encoder_model_type
        + "_"
        + args.decoder_model_type
        + "_fix"
        + str(args.fix_model)
        + "_lr"
        + str(args.learning_rate)
        + "_latent"
        + str(args.latent_size)
        + "_bs"
        + str(args.per_gpu_train_batch_size)
        + "_epoch"
        + str(args.num_train_epochs)
        + "_beta"
        + str(args.beta)
        + "_dim_target_kl"
        + str(args.dim_target_kl)
        + "_wl"
        + str(args.latent_loss_weight)
        + "_wg"
        + str(args.gap_loss_weight)
    )

    args.output_dir = os.path.join(args.output_dir, output_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # 将训练的参数写入文件
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as fw:
        fw.write(json.dumps(vars(args), indent=4))
        fw.flush()

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

    # Load Optimius pre-trained model and tokenizer
    checkpoint = None
    if args.use_pretrained_model:
        global_step = args.gloabl_step_eval
        output_full_dir = os.path.join(
            args.checkpoint_dir, "checkpoint-full-{}".format(global_step)
        )
        checkpoint = torch.load(os.path.join(output_full_dir, "training.bin"))

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

    # setattr(decoder_config, "latent_size", args.latent_size)
    model_decoder = decoder_model_class.from_pretrained(
        args.decoder_model_name_or_path,
        latent_size=args.latent_size,
        latent_as_gpt_emb=latent_as_gpt_emb,
        latent_as_gpt_memory=latent_as_gpt_memory,
    )

    if args.fix_model == 84:
        print("Initialize the Extra Layer.")
        model_decoder.transformer.change_order()

    special_tokens_dict = {
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
    }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "tokens to GPT2")
    model_decoder.resize_token_embeddings(len(tokenizer_decoder))
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == "<PAD>"

    model_vae = VAE(
        model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args
    )

    if args.use_pretrained_model:
        model_vae.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device)  #

    if args.local_rank == 0:
        torch.distributed.barrier()
        # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    ##############################
    # Training
    global_step = 0
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
            # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataloader = build_dataload_and_cache_examples(
            args, [tokenizer_encoder, tokenizer_decoder], evaluate=False
        )

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss, _ = train(
            args,
            train_dataloader,
            model_vae,
            tokenizer_encoder,
            tokenizer_decoder,
        )
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
