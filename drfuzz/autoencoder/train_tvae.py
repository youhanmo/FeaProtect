import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
from ctgan.tvae import TVAE
from util.struct_util import *
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
def get_tave(params):
    _, _, is_enum, _, _ = get_low_high(train_df, params.dataset)
    v2_col2idx = {str(item): idx for idx, item in enumerate(columns)}
    discrete_columns = [v2_col2idx[key] for key, value in is_enum.items() if value]
    tvae = TVAE(embedding_dim=params.embedding_dim,
                compress_dims=params.compress_dim,
                decompress_dims=params.decompress_dim,
                batch_size=30,
        epochs=params.epoch, cuda=torch.cuda.is_available())
    tvae.fit(X_scaled, discrete_columns)
    return tvae



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Experiments Script For vae")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--embedding_dim", type=int, default=128) 
    parser.add_argument("--compress_dim", type=tuple, default=(128, 128))
    parser.add_argument("--decompress_dim", type=tuple, default=(128, 128))
    parser.add_argument("--output_name", type=str, default='BM', help="output name")
    params = parser.parse_args()
    
    train_df = pd.read_csv(os.path.join('../../dataset', params.dataset, 'train.csv'))

    columns = [col for col in train_df.columns if col != params.output_name]
    # 数据处理
    label_encoder = LabelEncoder()
    X = train_df.drop([params.output_name], axis=1)
    y = train_df[params.output_name]
    y = label_encoder.fit_transform(y)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    print("embedding_dim: ", params.embedding_dim, "compress_dim: ", params.compress_dim, "decompress_dims: ", params.decompress_dim)

    model_name = 'tave'
    model_path = os.path.join('./model', params.dataset, model_name,
                                str(params.embedding_dim) + '_' + str(params.compress_dim[0]) + '_' + str(
                                    params.decompress_dim[0]))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model = get_tave(params)
    torch.save(model, os.path.join(model_path, 'model.h5'))