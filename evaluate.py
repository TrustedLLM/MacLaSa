import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from collections import Counter
from nltk.tokenize import word_tokenize
from transformers import set_seed, AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

root_dir = "/".join(os.path.dirname(os.path.abspath("__file__")).split("/")[:-1])

set_seed(42)

LABEL_MAP = {
    "imdb19": ["pos", "neg"],
    "imdb": ["pos", "neg"],
    "agnews": ["world", "sports", "business", "scitech"],
    "agnews50": ["world", "sports", "business", "scitech"],
    "yelp_senti": ["pos", "neg"],
    "yelp_topic": ["asian", "american", "mexican"],
    "yelp_cat": ["asian", "american", "mexican"],
}


def read_file(file_path):
    data = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            data.append(line.strip().lower())
    return data


class Evaluator:
    def __init__(
        self,
        eval_sentiment,
        sentiment_discriminator_name,
        sentiment_discriminator_save_dir,
        eval_topic,
        topic_discriminator_name,
        topic_discriminator_save_dir,
        eval_batch_size,
        max_len,
    ):
        self.eval_sentiment = eval_sentiment
        self.eval_topic = eval_topic
        self.sentiment_discriminator_name = sentiment_discriminator_name
        self.sentiment_discriminator_save_dir = sentiment_discriminator_save_dir
        self.topic_discriminator_name = topic_discriminator_name
        self.topic_discriminator_save_dir = topic_discriminator_save_dir
        self.eval_batch_size = eval_batch_size
        self.max_len = max_len

        self.load_discriminators()

    def load_discriminators(self, tokenizer_name="roberta-large"):
        # load discriminator if needed.

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(
                self.sentiment_discriminator_save_dir,
                self.sentiment_discriminator_name,
            )
        )

        if self.eval_sentiment:
            # load pretrained roberta classifier
            sentiment_discriminator_save_path = os.path.join(
                root_dir,
                os.path.join(
                    self.sentiment_discriminator_save_dir,
                    self.sentiment_discriminator_name,
                ),
            )
            print("Loading from checkpoint: ", sentiment_discriminator_save_path)
            self.sentiment_discriminator = (
                AutoModelForSequenceClassification.from_pretrained(
                    sentiment_discriminator_save_path, num_labels=2
                )
            )
            self.sentiment_discriminator.eval()
            self.sentiment_discriminator.to(device)

        if self.eval_topic:
            # load pre-trained roberta topic classifier
            topic_discriminator_path = os.path.join(
                root_dir,
                os.path.join(
                    self.topic_discriminator_save_dir,
                    self.topic_discriminator_name,
                ),
            )
            print("Loading from checkpoint: ", topic_discriminator_path)
            self.topic_discriminator = (
                AutoModelForSequenceClassification.from_pretrained(
                    topic_discriminator_path, num_labels=4
                )
            )
            self.topic_discriminator.eval()
            self.topic_discriminator.to(device)

    def fetch_label_index(self, output_filename):
        # 获取当前待评估文件所属数据集的标签数量，和当前待评估文件的期望标签
        dataset = output_filename.split("-")[0]
        polarity = output_filename.split("-")[1]
        n_labels = len(LABEL_MAP[dataset])
        label_index = LABEL_MAP[dataset].index(polarity)
        return n_labels, label_index

    def compute_multi_attributes_accuracy(
        self,
        sentences,
        target_senti_id,
        target_topic_id,
    ):
        # compute sentiment and topic control accuracy simultaneously
        correct_count = 0
        total_count = 0

        for i in range(0, len(sentences), self.eval_batch_size):
            with torch.no_grad():
                senti_logits = self.sentiment_discriminator(
                    **(
                        self.tokenizer(
                            sentences[i : i + self.eval_batch_size],
                            max_length=self.max_len,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                        )
                    ).to(device)
                )[0]
                topic_logits = self.topic_discriminator(
                    **(
                        self.tokenizer(
                            sentences[i : i + self.eval_batch_size],
                            max_length=self.max_len,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                        )
                    ).to(device)
                )[0]

            senti_preds = torch.argmax(senti_logits, dim=-1).cpu()
            topic_preds = torch.argmax(topic_logits, dim=-1).cpu()

            senti_labels = torch.LongTensor([target_senti_id] * senti_preds.size(0))
            topic_labels = torch.LongTensor([target_topic_id] * topic_preds.size(0))

            correct_count += (
                torch.logical_and(
                    senti_preds.eq(senti_labels), topic_preds.eq(topic_labels)
                )
                .sum()
                .item()
            )
            total_count += senti_preds.size(0)

        assert len(sentences) == total_count
        return correct_count / total_count * 100

    def compute_acc(self, tokenizer, model, sentences, label_index):
        # compute sentiment control accuracy
        correct_count = 0
        total_count = 0

        for i in range(0, len(sentences), self.eval_batch_size):
            with torch.no_grad():
                predict = model(
                    **(
                        tokenizer(
                            sentences[i : i + self.eval_batch_size],
                            max_length=self.max_len,
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt",
                        )
                    ).to(device)
                )[0]

            preds = torch.argmax(predict, dim=-1).cpu()
            labels = torch.LongTensor([label_index] * len(preds))
            correct_count += preds.eq(labels).sum().item()
            total_count += preds.size(0)

        assert len(sentences) == total_count
        return correct_count / total_count * 100

    def compute_topic_acc(self, generated_texts, label_index):
        return self.compute_acc(
            self.tokenizer, self.topic_discriminator, generated_texts, label_index
        )

    def compute_sentiment_acc(self, generated_texts, label_index):
        return self.compute_acc(
            self.tokenizer, self.sentiment_discriminator, generated_texts, label_index
        )

    def compute_ppl(self, sentences, stride=512):
        # load gpt2 model to compute ppl scores
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        model.to(device)

        encodings = tokenizer.encode("\n\n".join(sentences), return_tensors="pt")
        max_length = model.config.n_positions

        nlls = []
        for i in range(0, encodings.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.size(1))
            trg_len = end_loc - i
            input_ids = encodings[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            # target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                nll = outputs[0] * trg_len
            nlls.append(nll)
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return ppl.item()

    def compute_distinct(self, generated_sequences):
        """Calculate intra/inter distinct 1/2.
        Code from https://github.com/PaddlePaddle/models/blob/release/1.6/PaddleNLP/Research/Dialogue-PLATO/plato/metrics/metrics.py
        """

        tokenized_seqs = [word_tokenize(seq) for seq in generated_sequences]

        unigrams_all, bigrams_all = Counter(), Counter()

        for seq in tokenized_seqs:

            unigrams = Counter(seq)
            bigrams = Counter(zip(seq, seq[1:]))

            unigrams_all.update(unigrams)
            bigrams_all.update(bigrams)

        inter_dist1 = (len(unigrams_all) + 1e-12) / (sum(unigrams_all.values()) + 1e-5)
        inter_dist2 = (len(bigrams_all) + 1e-12) / (sum(bigrams_all.values()) + 1e-5)

        return inter_dist1, inter_dist2

    def evaluate(self, generated_texts, senti_label_index=None, topic_label_index=None):

        print("---------Evaluate Result------------")

        ppl = self.compute_ppl(generated_texts)
        print("PPL Score {:.2f}\n".format(ppl))

        dist1, dist2 = self.compute_distinct(generated_texts)
        print(
            "Distinct-1 Score: {:.2f}, Distinct-2 Score: {:.2f}\n".format(dist1, dist2)
        )

        if self.eval_sentiment and self.eval_topic:
            multi_attr_acc = self.compute_multi_attributes_accuracy(
                generated_texts,
                senti_label_index,
                topic_label_index,
            )
            print("Multi-Attributes Control Acc: {:.2f}\n".format(multi_attr_acc))

        print("------------------------------------")

    def evaluate_file(self, output_file_path):
        output_filename = os.path.split(output_file_path)[1]
        _, label_index = self.fetch_label_index(output_filename)
        generated_texts = read_file(output_file_path)
        if self.eval_sentiment:
            self.evaluate(generated_texts, senti_label_index=label_index)
        else:
            self.evaluate(generated_texts, topic_label_index=label_index)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_sentiment",
        action="store_true",
        help="whether to evaluate sentiment accuracy of generated texts",
    )
    parser.add_argument(
        "--sentiment_discriminator_name",
        type=str,
        default="imdb19_lr5e-05_warmup200",
        help="model name of pretrained sentiment discriminator",
    )
    parser.add_argument(
        "--sentiment_discriminator_save_dir",
        type=str,
        default="",
        help="path of pretrained sentiment discriminator",
    )
    parser.add_argument(
        "--eval_topic",
        action="store_true",
        help="whether to evaluate topic accuracy of generated texts",
    )
    parser.add_argument(
        "--topic_discriminator_name",
        type=str,
        # default="yelp_cat",
        default="agnews_lr5e-05_warmup200",
        help="model name of pretrained topic discriminator",
    )
    parser.add_argument(
        "--topic_discriminator_save_dir",
        type=str,
        default="",
        help="path of pretrained topic discriminator",
    )
    parser.add_argument("--eval_batch_size", type=int, default=50)
    parser.add_argument(
        "--output_save_dir",
        type=str,
        default="",
        help="path of generated texts",
    )
    parser.add_argument(
        "--saved_file",
        type=str,
        default="",
        help="file that need to be evaluated.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=50,
        help="Max Length of Generated Texts: 100(default)",
    )

    args = parser.parse_args()

    args.eval_sentiment = True
    args.eval_topic = True

    assert any(
        [args.eval_sentiment, args.eval_topic]
    ), "At least eval_sentiment or eval_topic should be True"

    print("*********Evaluating Params*************")
    if args.eval_sentiment:
        print("Sentiment Discriminator Name:", args.sentiment_discriminator_name)
        print(
            "Sentiment Discriminator Save Directory:",
            args.sentiment_discriminator_save_dir,
        )
    if args.eval_topic:
        print("Topic Discriminator Name:", args.topic_discriminator_name)
        print(
            "Topic Discriminator Save Directory:",
            args.topic_discriminator_save_dir,
        )
    print("Eval Batch Size:", args.eval_batch_size)
    print("Output Save Directory:", args.output_save_dir)
    print("***************************************")

    evaluator = Evaluator(
        args.eval_sentiment,
        args.sentiment_discriminator_name,
        args.sentiment_discriminator_save_dir,
        args.eval_topic,
        args.topic_discriminator_name,
        args.topic_discriminator_save_dir,
        args.eval_batch_size,
        args.max_len,
    )

    print(f"Enumerating output directory at {args.output_save_dir}")

    for saved_file in os.listdir(args.output_save_dir):
        print(f"Evaluating generated file: {saved_file}")

        sentiment_label_index = int(float(saved_file.split("_")[2]))
        topic_label_index = int(float(saved_file.split("_")[3].replace(".txt", "")))

        generated_texts = read_file(os.path.join(args.output_save_dir, saved_file))

        evaluator.evaluate(
            generated_texts,
            senti_label_index=sentiment_label_index,
            topic_label_index=topic_label_index,
        )
