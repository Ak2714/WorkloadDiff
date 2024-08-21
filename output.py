import re
import os
import pickle
import pandas as pd
import argparse
import time
parser = argparse.ArgumentParser(description="WorkloadDiff")
parser.add_argument("--record_root_path", type=str, default="")
parser.add_argument("--record_file_name", type=str, default="")


def extract_model (record_file_path):
    with open(record_file_path, 'r') as f:
        data = f.read()
    pattern = r'"model_folder":\s*"(.+?)"'
    match = re.findall(pattern, data)
    return match


def data_processing(model_lst, root_path):
    tag = ' '
    save_path = os.path.join(root_path, 'record')
    os.makedirs(save_path, exist_ok=True)

    label = ['RMSE', 'MAE', 'MAPE', 'R_2', 'CRPS', 'RMSLE', 'label']
    value_lst = []
    for model in model_lst:
        file_path = os.path.join(root_path, model, 'result_nsample10.pk')
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                data = [round(d, 4) for d in data]
                data.append(tag)
                value_lst.append(data)
        else:
            print("File does not exist：", file_path)

    df = pd.DataFrame(value_lst, columns=label)
    record_time = time.strftime('%Y-%m-%d_%H_%M_%S_', time.localtime())
    save_file_name = '_'.join([model.split('_')[-1] for model in model_lst])
    save_file_name = record_time + save_file_name

    df.to_excel(os.path.join(save_path, save_file_name + '.xlsx'), index=False)
    print(df)
    mean_v = df.drop(columns=['label', 'RMSLE']).mean()
    print(mean_v)
    ans = [' & ' + str(round(i, 4)) for i in mean_v.tolist()]
    print(''.join(ans) + ' \\\\')


# python output.py --record_root_path $cur_file_path --record_file_name ExperimentRecord.txt
if __name__ == '__main__':

    args = parser.parse_args()
    record_root_path = args.record_root_path
    record_file_name = args.record_file_name

    record_file_path = os.path.join(record_root_path, record_file_name)

    if record_file_path != "":
        model_lst = extract_model (record_file_path)
        data_processing(model_lst, record_root_path)
