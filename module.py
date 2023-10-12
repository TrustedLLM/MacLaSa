import torch
import torch.nn as nn
import numpy as np

import math


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


class VAE(nn.Module):
    """VAE with normal prior"""

    def __init__(self, encoder, decoder, tokenizer_encoder, tokenizer_decoder, args):  #
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.args = args
        self.nz = args.latent_size
        self.latent_size = args.latent_size
        self.latent_num = 1

        self.eos_token_id = tokenizer_decoder.convert_tokens_to_ids(
            [tokenizer_decoder.eos_token]
        )[0]
        self.pad_token_id = tokenizer_decoder.convert_tokens_to_ids(
            [tokenizer_decoder.pad_token]
        )[0]
        self.bos_token_id = tokenizer_decoder.convert_tokens_to_ids(
            [tokenizer_decoder.bos_token]
        )[0]
        self.tokenizer_decoder = tokenizer_decoder
        self.tokenizer_encoder = tokenizer_encoder
        # Standard Normal prior
        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)

        self.aspect_head_num = 2
        self.aspect_gap_loss_amplification = 5
        self.aspect_gap_head = [None for i in range(self.aspect_head_num)]

        self.set_latent_classify_head(
            head_num=2,
            class_num_per_head=[2, 4],
            mid_size=128,
        )

    def connect(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        # pdb.set_trace()
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)
        # z = mean
        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def connect_deterministic(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)

        mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        # pdb.set_trace()
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        logvar.fill_(0.0)
        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        pos_labels,
        head_index,
        std=False,
    ):

        attention_mask = (
            encoder_input_ids != self.tokenizer_encoder.pad_token_id
        ).float()

        reconstruction_mask = (
            decoder_input_ids != self.tokenizer_decoder.pad_token_id
        ).float()
        sent_length = torch.sum(reconstruction_mask, dim=1)

        outputs = self.encoder(encoder_input_ids, attention_mask)
        pooled_hidden_fea = outputs[1]

        latent_z = None

        if self.args.fb_mode == 0:
            # Connect hidden feature to the latent space
            latent_z, loss_kl = self.connect(pooled_hidden_fea)
            latent_z = latent_z.squeeze(1)

            # Decoding
            outputs = self.decoder(
                input_ids=decoder_input_ids,
                past=latent_z,
                labels=decoder_input_ids,
                label_ignore=self.pad_token_id,
            )
            loss_rec = outputs[
                0
            ]  # model outputs are always tuple in pytorch-transformers (see doc)

        elif self.args.fb_mode == 1:
            # Connect hidden feature to the latent space
            mu, logvar = self.encoder.linear(pooled_hidden_fea).chunk(2, -1)
            latent_z = self.reparameterize(mu, logvar, nsamples=1)
            latent_z = latent_z.squeeze(1)
            loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
            kl_mask = (loss_kl > self.args.dim_target_kl).float()
            loss_kl = (kl_mask * loss_kl).sum(dim=1)
            outputs = self.decoder(
                input_ids=decoder_input_ids,
                past=latent_z,
                labels=decoder_input_ids,
                label_ignore=self.pad_token_id,
            )
            loss_rec = outputs[
                0
            ]  # model outputs are always tuple in pytorch-transformers (see doc)

        elif self.args.fb_mode == 2:
            # Connect hidden feature to the latent space
            latent_z, loss_kl = self.connect_deterministic(pooled_hidden_fea)
            latent_z = latent_z.squeeze(1)
            outputs = self.decoder(
                input_ids=decoder_input_ids,
                past=latent_z,
                labels=decoder_input_ids,
                label_ignore=self.pad_token_id,
            )
            loss_rec = outputs[
                0
            ]  # model outputs are always tuple in pytorch-transformers (see doc)
        elif self.args.fb_mode == 3:  # For Autoencoder
            # Connect hidden feature to the latent space
            mu, logvar = self.encoder.linear(pooled_hidden_fea).chunk(2, -1)
            latent_z = mu.squeeze(1)
            loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
            outputs = self.decoder(
                input_ids=decoder_input_ids,
                past=latent_z,
                labels=decoder_input_ids,
                label_ignore=self.pad_token_id,
            )
            loss_rec = outputs[
                0
            ]  # model outputs are always tuple in pytorch-transformers (see doc)

        # latent classification loss
        latent_classify_loss = self.latent_classify_loss(
            latent_z, pos_labels, head_index
        )

        # latent gap loss
        gap_loss = self.aspect_gap_loss(latent_z, head_index)

        if self.args.length_weighted_loss:
            if self.args.beta == 0.0:
                loss = loss_rec / sent_length
            else:
                loss = (loss_rec + self.args.beta * loss_kl) / sent_length
        else:
            loss = loss_rec + self.args.beta * loss_kl

        # 权重系数还没确定
        loss += self.args.latent_loss_weight * latent_classify_loss
        if gap_loss is not None:
            loss += self.args.gap_loss_weight * gap_loss

        if std:
            return (
                loss_rec,
                loss_kl,
                latent_classify_loss,
                gap_loss,
                loss,
                latent_z.mean(),
                latent_z.std(),
            )
        else:
            return loss_rec, loss_kl, latent_classify_loss, gap_loss, loss

    def encode_x(self, inputs, repa=False):
        attention_mask = (inputs != self.tokenizer_encoder.pad_token_id).float()
        outputs = self.encoder(inputs, attention_mask)
        pooled_hidden_fea = outputs[
            1
        ]  # model outputs are always tuple in pytorch-transformers (see doc)

        # Connect hidden feature to the latent space
        mu, logvar = self.encoder.linear(pooled_hidden_fea).chunk(2, -1)
        if repa:
            latent_z = self.reparameterize(mu, logvar, nsamples=1)
            latent_z = latent_z.squeeze(1)
        else:
            latent_z = mu
        return latent_z

    def encoder_sample(self, bert_fea, nsamples):
        """sampling from the encoder
        Returns: Tensor1
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
        """

        # (batch_size, nz)

        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        mu, logvar = mu.squeeze(0), logvar.squeeze(0)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        return z, (mu, logvar)

    def encode_stats(self, x):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the mean of latent z with shape [batch, nz]
            Tensor2: the logvar of latent z with shape [batch, nz]
        """

        return self.encoder.encode_stats(x)

    def decode(self, z, strategy, K=10):
        """generate samples from z given strategy
        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """

        if strategy == "beam":
            return self.decoder.beam_search_decode(z, K)
        elif strategy == "greedy":
            return self.decoder.greedy_decode(z)
        elif strategy == "sample":
            return self.decoder.sample_decode(z)
        else:
            raise ValueError("the decoding strategy is not supported")

    def decode_eval_greedy(self, x, z):
        #         n_sample, length = x.size()
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = (
                x.unsqueeze(1)
                .repeat(1, z_shape[1], 1)
                .contiguous()
                .view(x_shape[0] * z_shape[1], x_shape[-1])
            )
            z = z.contiguous().view(x_shape[0] * z_shape[1], z_shape[-1])
        batch_size = z.size()[0]
        decoded_batch = [[] for _ in range(batch_size)]
        x_ = torch.zeros_like(z[:, :1], dtype=torch.long) + self.bos_token_id
        #         for i in range(length):
        mask = torch.zeros_like(z[:, 0], dtype=torch.long) + 1
        length_c = 1
        end_symbol = torch.zeros_like(mask, dtype=torch.long) + self.eos_token_id
        while mask.sum().item() != 0 and length_c < 100:
            output = self.decoder(input_ids=x_, past=z)
            out_token = output[0][:, -1:].max(-1)[1]
            x_ = torch.cat((x_, out_token), -1)
            length_c += 1
            mask = torch.mul((out_token.squeeze(-1) != end_symbol), mask)
            for i in range(batch_size):
                #                 word = self.tokenizer_decoder.decode(out_token[i].tolist())
                if mask[i].item():
                    decoded_batch[i].append(
                        self.tokenizer_decoder.decode(out_token[i].item())
                    )
        #         out_tokens = x_[:,1:]
        for i in range(batch_size):
            decoded_batch[i] = "".join(decoded_batch[i])
        return decoded_batch

    def decode_eval_greedy_tf(self, x, z):
        #         n_sample, length = x.size()
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = (
                x.unsqueeze(1)
                .repeat(1, z_shape[1], 1)
                .contiguous()
                .view(x_shape[0] * z_shape[1], x_shape[-1])
            )
            z = z.contiguous().view(x_shape[0] * z_shape[1], z_shape[-1])
        batch_size = z.size()[0]
        decoded_batch = [[] for _ in range(batch_size)]
        x_ = torch.zeros_like(z[:, :1], dtype=torch.long) + self.bos_token_id
        #         for i in range(length):
        mask = torch.zeros_like(z[:, 0], dtype=torch.long) + 1
        length_c = 1
        end_symbol = torch.zeros_like(mask, dtype=torch.long) + self.eos_token_id
        while mask.sum().item() != 0 and length_c < 100 and length_c <= x_shape[-1]:
            output = self.decoder(input_ids=x_, past=z)
            out_token = output[0][:, -1:].max(-1)[1]
            x_ = torch.cat((x_, x[:, length_c : length_c + 1]), -1)
            length_c += 1
            mask = torch.mul((out_token.squeeze(-1) != end_symbol), mask)
            for i in range(batch_size):
                #                 word = self.tokenizer_decoder.decode(out_token[i].tolist())
                if mask[i].item():
                    decoded_batch[i].append(
                        self.tokenizer_decoder.decode(out_token[i].item())
                    )
        #         out_tokens = x_[:,1:]
        for i in range(batch_size):
            decoded_batch[i] = "".join(decoded_batch[i])
        return decoded_batch

    def reconstruct(self, x, decoding_strategy="greedy", K=5):
        """reconstruct from input x
        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter
        Returns: List1
            List1: a list of decoded word sequence
        """
        z = self.sample_from_inference(x).squeeze(1)

        return self.decode(z, decoding_strategy, K)

    def log_probability(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
        #         outputs_ = self.decode_eval_gy(x,z)
        outputs = self.decoder(
            input_ids=x, past=z, labels=x, label_ignore=self.pad_token_id
        )
        loss_rec = outputs[0]
        return -loss_rec

    def log_probability_out(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
        #         outputs_ = self.decode_eval_gy(x,z)
        outputs = self.decoder(
            input_ids=x, past=z, labels=x, label_ignore=self.pad_token_id
        )
        return outputs

    def loss_iw(self, x0, x1, nsamples=50, ns=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """

        # encoding into bert features
        bert_fea = self.encoder(x0)[1]
        # (batch_size, nz)

        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

        ##################
        # compute KL
        ##################
        # pdb.set_trace()
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        # mu, logvar = mu.squeeze(0), logvar.squeeze(0)
        ll_tmp, rc_tmp = [], []
        for _ in range(int(nsamples / ns)):
            # (batch, nsamples, nz)
            z = self.reparameterize(mu, logvar, ns)
            # past = self.decoder.linear(z)
            past = z

            # [batch, nsamples]
            log_prior = self.eval_prior_dist(z)
            log_gen = self.eval_cond_ll(x1, past)  # given z, the prob of x
            log_infer = self.eval_inference_dist(z, (mu, logvar))

            # pdb.set_trace()
            log_gen = log_gen.unsqueeze(0).contiguous().view(z.shape[0], -1)

            # pdb.set_trace()
            rc_tmp.append(log_gen)
            ll_tmp.append(log_gen + log_prior - log_infer)

        log_prob_iw = log_sum_exp(torch.cat(ll_tmp, dim=-1), dim=-1) - math.log(
            nsamples
        )
        log_gen_iw = torch.mean(torch.cat(rc_tmp, dim=-1), dim=-1)

        return log_prob_iw, log_gen_iw, KL

    def rec_sample(self, x0, x1, sample=False):
        bert_fea = self.encoder(x0)[1]
        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        z = self.reparameterize(mu, logvar, 1)
        rec = "rec"
        if sample:
            z = (
                torch.tensor(
                    np.random.normal(size=(z.size()[0], z.size()[-1])),
                    dtype=torch.double,
                )
                .cuda()
                .unsqueeze(1)
            )
            x1 = torch.zeros_like(x1)
            rec = "sample"
        decoded_batch = self.decode_eval_greedy(x1, z)
        with open(
            "/home/lptang/Optimus/samples/"
            + self.args.output_dir.split("/")[-1]
            + "."
            + str(self.args.gloabl_step_eval)
            + "."
            + rec,
            "a+",
        ) as f:
            for sent in decoded_batch:
                f.write(sent + "\n")

    def nll_iw(self, x0, x1, nsamples, ns=1):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x0, x1:  two different tokenization results of x, where x is the data tensor with shape (batch, *).
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """

        # compute iw every ns samples to address the memory issue
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10

        # TODO: note that x is forwarded twice in self.encoder.sample(x, ns) and self.eval_inference_dist(x, z, param)
        # .      this problem is to be solved in order to speed up

        tmp = []
        for _ in range(int(nsamples / ns)):
            # [batch, ns, nz]

            # Chunyuan:
            # encoding into bert features
            pooled_hidden_fea = self.encoder(x0)[1]

            # param is the parameters required to evaluate q(z|x)
            z, param = self.encoder_sample(pooled_hidden_fea, ns)

            # [batch, ns]
            log_comp_ll = self.eval_complete_ll(x1, z)
            log_infer_ll = self.eval_inference_dist(z, param)

            tmp.append(log_comp_ll - log_infer_ll)

        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)

        return ll_iw

    def KL(self, x):
        _, KL = self.encode(x, 1)

        return KL

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """

        # (k^2)
        return self.prior.log_prob(zrange).sum(dim=-1)

    def eval_complete_ll(self, x, z):
        """compute log p(z,x)
        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
        """

        # [batch, nsamples]
        log_prior = self.eval_prior_dist(z)
        log_gen = self.eval_cond_ll(x, z)

        return log_prior + log_gen

    def eval_cond_ll(self, x, z):
        """compute log p(x|z)"""
        x_shape = list(x.size())
        z_shape = list(z.size())
        if len(z_shape) == 3:
            x = (
                x.unsqueeze(1)
                .repeat(1, z_shape[1], 1)
                .contiguous()
                .view(x_shape[0] * z_shape[1], x_shape[-1])
            )
            z = z.contiguous().view(x_shape[0] * z_shape[1], z_shape[-1])

        return self.log_probability(x, z)

    def eval_log_model_posterior(self, x, grid_z):
        """perform grid search to calculate the true posterior
         this function computes p(z|x)
        Args:
            grid_z: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/pace
        Returns: Tensor
            Tensor: the log posterior distribution log p(z|x) with
                    shape [batch_size, K^2]
        """
        try:
            batch_size = x.size(0)
        except:
            batch_size = x[0].size(0)

        # (batch_size, k^2, nz)
        grid_z = grid_z.unsqueeze(0).expand(batch_size, *grid_z.size()).contiguous()

        # (batch_size, k^2)
        log_comp = self.eval_complete_ll(x, grid_z)

        # normalize to posterior
        log_posterior = log_comp - log_sum_exp(log_comp, dim=1, keepdim=True)

        return log_posterior

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        bert_fea = self.encoder(x)[1]
        mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)
        #         z, _ = self.encoder.sample(x, nsamples)
        z = self.reparameterize(mu, logvar, nsamples)
        return z

    def sample_from_posterior(self, x, nsamples):
        """perform MH sampling from model posterior
        Returns: Tensor
            Tensor: samples from model posterior with
                shape (batch_size, nsamples, nz)
        """

        # use the samples from inference net as initial points
        # for MCMC sampling. [batch_size, nsamples, nz]
        cur = self.encoder.sample_from_inference(x, 1)
        cur_ll = self.eval_complete_ll(x, cur)
        total_iter = self.args.mh_burn_in + nsamples * self.args.mh_thin
        samples = []
        for iter_ in range(total_iter):
            next = torch.normal(
                mean=cur, std=cur.new_full(size=cur.size(), fill_value=self.args.mh_std)
            )
            # [batch_size, 1]
            next_ll = self.eval_complete_ll(x, next)
            ratio = next_ll - cur_ll

            accept_prob = torch.min(ratio.exp(), ratio.new_ones(ratio.size()))

            uniform_t = accept_prob.new_empty(accept_prob.size()).uniform_()

            # [batch_size, 1]
            mask = (uniform_t < accept_prob).float()
            mask_ = mask.unsqueeze(2)

            cur = mask_ * next + (1 - mask_) * cur
            cur_ll = mask * next_ll + (1 - mask) * cur_ll

            if (
                iter_ >= self.args.mh_burn_in
                and (iter_ - self.args.mh_burn_in) % self.args.mh_thin == 0
            ):
                samples.append(cur.unsqueeze(1))

        return torch.cat(samples, dim=1)

    def calc_model_posterior_mean(self, x, grid_z):
        """compute the mean value of model posterior, i.e. E_{z ~ p(z|x)}[z]
        Args:
            grid_z: different z points that will be evaluated, with
                    shape (k^2, nz), where k=(zmax - zmin)/pace
            x: [batch, *]
        Returns: Tensor1
            Tensor1: the mean value tensor with shape [batch, nz]
        """

        # [batch, K^2]
        log_posterior = self.eval_log_model_posterior(x, grid_z)
        posterior = log_posterior.exp()

        # [batch, nz]
        return torch.mul(posterior.unsqueeze(2), grid_z.unsqueeze(0)).sum(1)

    def calc_infer_mean(self, x):
        """
        Returns: Tensor1
            Tensor1: the mean of inference distribution, with shape [batch, nz]
        """

        mean, logvar = self.encoder.forward(x)

        return mean

    def eval_inference_dist(self, z, param):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)
        mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev**2) / var).sum(dim=-1) - 0.5 * (
            nz * math.log(2 * math.pi) + logvar.sum(-1)
        )

        return log_density

    def calc_mi(self, test_data_batch, args):
        mi = 0
        num_examples = 0

        mu_batch_list, logvar_batch_list = [], []
        neg_entropy = 0.0
        for batch_data in test_data_batch:
            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            # (batch_size, nz)
            mu, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

            x_batch, nz = mu.size()

            # print(x_batch, end=' ')

            num_examples += x_batch

            # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)

            neg_entropy += (
                (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1))
                .sum()
                .item()
            )
            mu_batch_list += [mu.cpu()]
            logvar_batch_list += [logvar.cpu()]

        #             pdb.set_trace()

        neg_entropy = neg_entropy / num_examples
        ##print()

        num_examples = 0
        log_qz = 0.0
        for i in range(len(mu_batch_list)):
            ###############
            # get z_samples
            ###############
            mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()

            # [z_batch, 1, nz]

            z_samples = self.reparameterize(mu, logvar, 1)

            z_samples = z_samples.view(-1, 1, nz)
            num_examples += z_samples.size(0)

            ###############
            # compute density
            ###############
            # [1, x_batch, nz]
            # mu, logvar = mu_batch_list[i].cuda(), logvar_batch_list[i].cuda()
            # indices = list(np.random.choice(np.arange(len(mu_batch_list)), 10)) + [i]
            indices = np.arange(len(mu_batch_list))
            mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).cuda()
            logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).cuda()
            x_batch, nz = mu.size()

            mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
            var = logvar.exp()

            # (z_batch, x_batch, nz)
            dev = z_samples - mu

            # (z_batch, x_batch)
            log_density = -0.5 * ((dev**2) / var).sum(dim=-1) - 0.5 * (
                nz * math.log(2 * math.pi) + logvar.sum(-1)
            )

            # log q(z): aggregate posterior
            # [z_batch]
            log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

        log_qz /= num_examples
        mi = neg_entropy - log_qz

        return mi

    def calc_au(self, eval_dataloader, args, delta=0.01):
        """compute the number of active units"""
        cnt = 0
        for batch_data in eval_dataloader:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            # (batch_size, nz)
            mean, logvar = self.encoder.linear(bert_fea).chunk(2, -1)

            if cnt == 0:
                means_sum = mean.sum(dim=0, keepdim=True)
            else:
                means_sum = means_sum + mean.sum(dim=0, keepdim=True)
            cnt += mean.size(0)

        # (1, nz)
        mean_mean = means_sum / cnt

        cnt = 0
        for batch_data in eval_dataloader:

            x0, _, _ = batch_data
            x0 = x0.to(args.device)

            # encoding into bert features
            bert_fea = self.encoder(x0)[1]

            # (batch_size, nz)
            mean, _ = self.encoder.linear(bert_fea).chunk(2, -1)

            if cnt == 0:
                var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
            else:
                var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
            cnt += mean.size(0)

        # (nz)
        au_var = var_sum / (cnt - 1)

        return (au_var >= delta).sum().item(), au_var

    def set_latent_classify_head(self, head_num=1, class_num_per_head=2, mid_size=128):
        if type(class_num_per_head) is list:
            self.latent_classify_head = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.latent_num * self.latent_size, mid_size),
                        nn.ReLU(),
                        nn.Linear(mid_size, head_num),
                    )
                    for head_num in class_num_per_head
                ]
            )
        else:
            self.latent_classify_head = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.latent_num * self.latent_size, mid_size),
                        nn.ReLU(),
                        nn.Linear(mid_size, class_num_per_head),
                    )
                    for i in range(head_num)
                ]
            )

    def set_aspect_gap_head(self, latent, head_index):
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)

        if len(latent.shape) == 2:
            mean_latent = torch.mean(latent.detach(), dim=0)
            if self.aspect_gap_head[head_index] is not None:
                assert self.aspect_gap_head[head_index].shape == mean_latent.shape
            self.aspect_gap_head[head_index] = mean_latent
        elif len(latent.shape) == 1:
            if self.aspect_gap_head[head_index] is not None:
                assert self.aspect_gap_head[head_index].shape == latent.shape
            self.aspect_gap_head[head_index] = latent.detach()

    def latent_classify_loss(self, latent, pos_label, head_index=None):
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)

        if head_index is None:
            print(
                "UserWarning: head_index not set for multiple classifier head, default to 0"
            )
            head_index = 0
        device = latent.device
        logits = self.latent_classify_head[head_index](latent)
        loss = torch.nn.functional.cross_entropy(logits, pos_label.to(device))
        return loss

    def aspect_gap_loss(self, latent, head_index):
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)

        mean_latent = torch.mean(latent, dim=0)
        loss = None
        for i in range(self.aspect_head_num):
            if i != head_index and self.aspect_gap_head[i] is not None:
                if loss is None:
                    loss = (
                        torch.nn.functional.mse_loss(
                            mean_latent, self.aspect_gap_head[i]
                        )
                        * self.aspect_gap_loss_amplification
                    )
                else:
                    loss += (
                        torch.nn.functional.mse_loss(
                            mean_latent, self.aspect_gap_head[i]
                        )
                        * self.aspect_gap_loss_amplification
                    )
        self.set_aspect_gap_head(mean_latent, head_index)
        return loss


class GAN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nz = args.latent_size
        self.args = args
        # self.latent_generator = nn.Linear(self.nz, self.nz)
        # self.latent_discriminator = nn.Linear(self.nz, 1)

        if (
            "large" in args.decoder_model_name_or_path
        ):  # and 'commongen' not in args.output_dir:
            self.latent_generator = nn.Sequential(
                nn.Linear(self.nz, self.nz),
                nn.Linear(self.nz, self.nz),
            )
            print("\n2 large generator\n")
        else:
            self.latent_generator = nn.Sequential(
                nn.Linear(self.nz, self.nz),
            )
            # self.latent_generator = nn.Sequential(
            #     nn.Linear(self.nz, self.nz*2),
            #     nn.LeakyReLU(),
            #     nn.Linear(self.nz*2, self.nz),
            #     nn.LeakyReLU(),
            #     nn.Linear(self.nz, self.nz)
            # )
        # if "news" in args.output_dir:
        self.latent_generator = nn.Sequential(
            nn.Linear(self.nz, self.nz),
            nn.Linear(self.nz, self.nz),
        )
        print("news in")
        self.latent_discriminator = nn.Linear(self.nz, 1)
        # self.latent_discriminator = nn.Sequential(
        #     nn.Linear(self.nz, self.nz),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.nz, 1))

        # self.latent_generator = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(self.nz, self.nz),
        #     nn.Dropout(0.5),
        #     nn.Linear(self.nz, self.nz),
        # )
        #
        # self.latent_discriminator = nn.Sequential(
        #     nn.Linear(self.nz, self.nz//2),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Tanh(),
        #     nn.Linear(self.nz//2, 1))
        # self.latent_discriminator = nn.Sequential(
        #     nn.Linear(self.nz, self.nz),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Linear(self.nz, self.nz),
        #     # nn.LeakyReLU(0.2),
        #     nn.Linear(self.nz, 1),
        # )
        self.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

    def generate_z(self, bz, eval=False):
        random_noise = torch.FloatTensor(bz, self.nz).normal_(0, 1).to(self.args.device)
        gen_z = self.latent_generator(random_noise)
        # gen_z = random_noise
        # random_noise = torch.nn.init.normal_(
        #     torch.empty(bz, self.nz)).to(device=self.args.device, dtype=torch.float32)
        ###################
        # mu, logvar = self.latent_generator(random_noise).chunk(2, -1)
        # gen_z =self.reparameterize(mu,logvar).squeeze(1)
        #################
        # if eval:
        #     return mu
        return gen_z

    def discriminate(self, latent_z):
        bz = latent_z.size(0)
        gen_z = self.generate_z(bz)
        prob_encode_z_dis = self.latent_discriminator(latent_z).squeeze(1).float()
        prob_gen_z_dis = self.latent_discriminator(gen_z).squeeze(1).float()
        ones_label = torch.ones(bz).to(dtype=torch.float32, device=self.args.device)
        zeros_label = torch.zeros(bz).to(dtype=torch.float32, device=self.args.device)
        acc_gen = self.BCEWithLogitsLoss(prob_gen_z_dis, zeros_label).mean()
        acc_enc = self.BCEWithLogitsLoss(prob_encode_z_dis, ones_label).mean()
        return acc_gen, acc_enc, (acc_gen - acc_enc).abs()

    def discriminate_acc(self, latent_z):
        bz = latent_z.size(0)
        gen_z = self.generate_z(bz)
        prob_encode_z_dis = self.latent_discriminator(latent_z).squeeze(1).float()
        prob_gen_z_dis = self.latent_discriminator(gen_z).squeeze(1).float()
        gan_acc = ((prob_encode_z_dis >= 0).sum() + (prob_gen_z_dis < 0).sum()) / (
            2 * bz
        )
        # ones_label = torch.ones(bz).to(dtype=torch.float32, device=self.args.device)
        # zeros_label = torch.zeros(bz).to(dtype=torch.float32, device=self.args.device)
        # acc_gen = self.BCEWithLogitsLoss(prob_gen_z_dis, zeros_label).mean()
        # acc_enc = self.BCEWithLogitsLoss(prob_encode_z_dis, ones_label).mean()
        return gan_acc.item()

    def d_loss(self, latent_z):
        bz = latent_z.size(0)
        ones_label = torch.ones(bz).to(dtype=torch.float32, device=self.args.device)
        zeros_label = torch.zeros(bz).to(dtype=torch.float32, device=self.args.device)
        gen_z = self.generate_z(bz)
        prob_encode_z_dis = self.latent_discriminator(latent_z).squeeze(1).float()

        prob_gen_z_dis = self.latent_discriminator(gen_z).squeeze(1).float()
        loss_lsd = self.BCEWithLogitsLoss(
            prob_gen_z_dis, zeros_label
        ) + self.BCEWithLogitsLoss(prob_encode_z_dis, ones_label)
        # import pdb
        # pdb.set_trace()
        # norm_z = latent_z.norm(dim=-1).mean()
        # norm_gene = gen_z.norm(dim=-1).mean()

        return loss_lsd.mean()  # + torch.relu(0.95*norm_z - norm_gene)

    def g_loss(self, latent_z):
        bz = latent_z.size(0)
        ones_label = torch.ones(bz).to(dtype=torch.float32, device=self.args.device)
        gen_z = self.generate_z(bz)
        prob_gen_z_dis = self.latent_discriminator(gen_z).squeeze(1).float()
        loss_lsg = self.BCEWithLogitsLoss(prob_gen_z_dis, ones_label)

        #
        # loss_kl = 0.5 * (gen_z.pow(2) + logvar.exp() - logvar - 1)
        # kl_mask = (loss_kl > self.args.dim_target_kl).float()
        # loss_kl = (kl_mask * loss_kl).sum(dim=1).mean()
        return loss_lsg

    def forward(self, latent_z):
        bz = latent_z.size(0)
        ones_label = torch.ones(bz).to(dtype=torch.float32, device=self.args.device)
        zeros_label = torch.zeros(bz).to(dtype=torch.float32, device=self.args.device)
        gen_z = self.generate_z(bz)
        prob_encode_z_dis = self.latent_discriminator(latent_z).squeeze(1).float()
        prob_gen_z_dis = self.latent_discriminator(gen_z).squeeze(1).float()
        loss_lsd = self.BCEWithLogitsLoss(
            prob_gen_z_dis, zeros_label
        ) + self.BCEWithLogitsLoss(prob_encode_z_dis, ones_label)
        loss_lsg = self.BCEWithLogitsLoss(prob_gen_z_dis, ones_label)
        loss = loss_lsd + loss_lsg
        return loss.mean(), loss_lsd.mean(), loss_lsg.mean()


class DenseEmbedder(nn.Module):
    """Supposed to map small-scale features (e.g. labels) to some given latent dim"""

    def __init__(
        self, input_dim, up_dim, depth=4, num_classes=10, given_dims=None, norm=None
    ):
        super().__init__()
        self.net = nn.ModuleList()
        dims = np.linspace(input_dim, up_dim, depth).astype(int)

        for l in range(len(dims) - 1):
            self.net.append(nn.Dropout(0.2))
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))
            self.net.append(nn.LeakyReLU(0.2))

        self.last_dim = up_dim
        self.linear = nn.Linear(up_dim, num_classes)
        self.energy_weight = 1

    def set_energy_weight(self, weight):
        self.energy_weight = weight

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, :, None, None]

        for layer in self.net:
            x = layer(x)

        out = x.squeeze(-1).squeeze(-1)
        out = self.linear(out)
        logits = out
        return logits


class CCF(nn.Module):
    def __init__(self, classifier):
        super(CCF, self).__init__()
        self.f = nn.ModuleList()
        for cls in classifier:
            self.f.append(cls)

    def get_cond_energy(self, z, y_):
        energy_outs = []
        # for i, cls in enumerate(self.f):
        for i in range(y_.shape[1]):
            cls = self.f[i]
            logits = cls(z)
            # logits_list.append(logits)
            n_classes = logits.size(1)
            if n_classes > 1:
                y = y_[:, i].long()
                sigle_energy = torch.gather(
                    logits, 1, y[:, None]
                ).squeeze() - logits.logsumexp(1)
                energy_outs.append(cls.energy_weight * sigle_energy)
            else:
                assert n_classes == 1, n_classes
                y = y_[:, i].float()
                sigma = 0.1  # this value works well
                sigle_energy = (
                    -torch.norm(logits - y[:, None], dim=1) ** 2 * 0.5 / (sigma**2)
                )
                energy_outs.append(cls.energy_weight * sigle_energy)

        energy_output = torch.stack(energy_outs).sum(dim=0)
        return energy_output

    def get_cond_energy_single(self, z, y_):
        for i in range(y_.shape[1]):
            energy_outs = []
            cls = self.f[i]
            logits = cls(z)
            n_classes = logits.size(1)
            if n_classes > 1:
                y = y_[:, i].long()
                sigle_energy = torch.gather(
                    logits, 1, y[:, None]
                ).squeeze() - logits.logsumexp(1)
                energy_outs.append(cls.energy_weight * sigle_energy)
            else:
                assert n_classes == 1, n_classes
                y = y_[:, i].float()
                sigma = 0.1  # this value works well
                sigle_energy = (
                    -torch.norm(logits - y[:, None], dim=1) ** 2 * 0.5 / (sigma**2)
                )
                energy_outs.append(cls.energy_weight * sigle_energy)
        # print('dog:', round(energy_outs[0].sum().item(),2), '\tchild:', round(energy_outs[1].sum().item(),2), '\tball:',round(energy_outs[2].sum().item(),2))

        energy_output = torch.stack(energy_outs).sum(dim=0)
        return energy_output

    def forward(self, z, y):
        energy_output = self.get_cond_energy(z, y) - torch.norm(z, dim=1) ** 2 * 0.5
        return energy_output
