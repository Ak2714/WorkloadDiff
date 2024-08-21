import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI


class WorkloadDiff_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = False
        self.target_strategy = config["model"]["target_strategy"]

        config_diff = config["diffusion"]
        self.diffmodel = diff_CSDI(config_diff)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)


    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)

        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def calc_loss_valid(
            self, observed_data, cond_mask, observed_mask, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
            self, observed_data, cond_mask, observed_mask, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        current_alpha = self.alpha_torch[t]  # (B,1,1)

        # Generate random noise.
        noise = torch.randn_like(observed_data)

        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input)  # (B,K,L)
        target_mask = observed_mask - cond_mask

        # calculate loss
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_data.unsqueeze(1)], dim=1)  # (B,2,K,L)
        return total_input

    def undo(self, t, current_sample):
        t_shift = 0
        t = t + t_shift
        current_alpha = self.alpha_hat[t]  # (B,1,1)
        noise = torch.randn_like(current_sample)
        current_sample = (current_alpha ** 0.5) * current_sample + (1.0 - current_alpha) ** 0.5 * noise
        return current_sample

    def forward_diffuse(self, observed_data, t):
        # x_0 -> x_t
        noise = torch.randn_like(observed_data)
        current_alpha = self.alpha_torch[t]
        # x_{t-1}^{cond}
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        return noisy_data

    # reverse sampling
    def reverse_sample(self, current_sample, observed_data, cond_mask, t):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        diff_input = torch.cat([cond_obs, current_sample.unsqueeze(1)], dim=1)  # (B,2,K,L)
        # predictive noise
        predicted = self.diffmodel(diff_input)

        # x_{t-1}^{pred}
        current_sample = 1 / self.alpha_hat[t] ** 0.5 * (
                current_sample - (1 - self.alpha_hat[t]) / (1 - self.alpha_hat[t]) ** 0.5 * predicted)

        if t > 0:
            noise = torch.randn_like(current_sample)
            sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
            current_sample += sigma * noise
        return current_sample

    def resample(self, observed_data, t, cond_mask, current_sample):
        cond = self.forward_diffuse(observed_data, t)
        noise = self.reverse_sample(current_sample, observed_data, cond_mask, t)

        # Merge the two parts into a complete sequence.
        current_sample = cond * cond_mask + noise * (1 - cond_mask)
        return current_sample

    def impute(self, observed_data, cond_mask, n_samples, u_number):
        is_resample = True
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)
            # 49 -> 0
            for t in range(self.num_steps - 1, -1, -1):
                if is_resample:
                    for u in range(1, u_number + 1, 1):
                        # perform resampling
                        current_sample = self.resample(observed_data, t, cond_mask, current_sample)
                        if t > 0 and u < u_number:
                            current_sample = self.undo(t, current_sample)
                else:
                    current_sample = self.reverse_sample(current_sample, observed_data, cond_mask, t)
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        cond_mask = gt_mask
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        ans = loss_func(observed_data, cond_mask, observed_mask, is_train)
        return ans

    def evaluate(self, batch, n_samples, u_number):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            # randomly generate samples
            samples = self.impute(observed_data, cond_mask, n_samples, u_number)
            # to avoid double evaluation
            for i in range(len(cut_length)):
                target_mask[i, ..., 0: cut_length[i].item()] = 0

        return samples, observed_data, target_mask, observed_mask, observed_tp


class WorkloadDiff(WorkloadDiff_base):

    def __init__(self, config, device):
        feature_num = config['others']['feature_num']
        super(WorkloadDiff, self).__init__(feature_num, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
