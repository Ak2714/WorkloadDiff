import os.path

import numpy as np
import torch
import yaml
from torch.optim import Adam
from tqdm import tqdm
import pickle
from sklearn.metrics import r2_score

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

def train(model, config, train_loader, foldername="", ):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"
    else:
        output_path = './'

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    train_loss_lst = []
    all_epoch_tarin_loss_lst = []
    all_epoch_val_loss_lst = []

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                train_loss_lst.append(loss.item())
            lr_scheduler.step()

        all_epoch_tarin_loss_lst.append(train_loss_lst)
        train_loss_lst = []
    torch.save(model.state_dict(), output_path)

    with open(os.path.join(foldername, 'train_val_loss.pk'), 'wb') as f:
        pickle.dump({'train_loss': all_epoch_tarin_loss_lst, 'val_loss': all_epoch_val_loss_lst}, f)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j: j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", config=None):
    if config is None:
        path = "config/base.yaml"
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    u_number = config['others']['u_number']
    feature_num = config['others']['feature_num']

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        mape_total = 0
        rmsle_total = 0
        evalpoints_total = 0

        pre_lst = []
        tar_lst = []
        mean_r2_lst = []

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        source = 'target_predict_data.pickle'

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample, u_number)
                samples, c_target, eval_points, observed_points, observed_time = output

                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                # mse
                mse_current = (((samples_median.values - c_target) * eval_points) ** 2) * (scaler ** 2)

                # mae
                mae_current = (torch.abs((samples_median.values - c_target) * eval_points)) * scaler

                target_eval = c_target[eval_points == 1.].reshape(-1, feature_num)
                predict_eval = samples_median.values[eval_points == 1.].reshape(-1, feature_num)

                if True in np.isnan(predict_eval.cpu()):
                    print("The results contain null values, the program has stopped.")
                    exit(0)

                pre_lst.append(predict_eval)
                tar_lst.append(target_eval)
                mean_r2_lst.append(r2_score(target_eval.cpu(), predict_eval.cpu()))

                # mape
                mape_current = (
                    torch.abs((samples_median.values - c_target) / c_target * eval_points)
                )

                mape_current = torch.where(torch.isnan(mape_current), torch.full_like(mape_current, 0), mape_current)
                mape_current = torch.where(torch.isinf(mape_current), torch.full_like(mape_current, 0), mape_current)

                # rmsle
                rmsle_current = ((torch.log(c_target * eval_points + 1) - torch.log(samples_median.values * eval_points + 1)) ** 2 * eval_points)

                # Record all prediction results.
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                mape_total += mape_current.sum().item()
                rmsle_total += rmsle_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "mape_total": mape_total / evalpoints_total,
                        "rmsle_total": np.sqrt(rmsle_total / evalpoints_total),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                    foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            # crps
            CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

            y_target = torch.cat(tar_lst, dim=0).cpu()
            y_predict = torch.cat(pre_lst, dim=0).cpu()
            r_2 = r2_score(y_target, y_predict)

            with open(
                    foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        mape_total / evalpoints_total,
                        r_2,
                        CRPS,
                        np.sqrt(rmsle_total / evalpoints_total)
                    ],
                    f,
                )

            lst = [('key', 'value'), ('RMSE', np.sqrt(mse_total / evalpoints_total)),
                   ('MAE', mae_total / evalpoints_total),
                   ('MAPE', mape_total / evalpoints_total), ('R_2', r_2), ('CRPS', CRPS),
                   ('RMSLE', np.sqrt(rmsle_total / evalpoints_total)),
                   ('epoch', config['train']['epochs']), ('dir', config['others']['dir_dataset']),
                   ('model_folder', config['others']['model_folder'])]

            for i, j in lst:
                print('{:^15}{:^10}'.format(i, j))  # 居中对齐

            # Save the results as a file.
            with open(os.path.join(foldername, source), 'wb') as f:
                pickle.dump({'target': torch.cat(tar_lst, dim=0).cpu(), 'predict': torch.cat(pre_lst, dim=0).cpu()}, f)
