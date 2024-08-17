import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import net_model.pd_transformer_net as t

if __name__ == '__main__':
    bs = 10000
    features_tr_N, _ = t.get_data_("./data/datasets/train_set_people.npy", NP="N",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_va, _ = t.get_data_("./data/datasets/va_set_people.npy", NP="N",
                                 sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_te, label_te = t.get_data_("./data/datasets/test_set_people.npy", NP="N",
                                        sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    source_tr_N = features_tr_N.to(t.device)
    source_va = features_va.to(t.device)
    source_te = features_te.to(t.device)

    mask = t.shape_mask(source_te.shape[1] - 1, source_te.shape[1] - 1, k=1).to(t.device)
    #
    N_transformer_model = t.MakeTransformerOri(max_len=source_te.shape[1], e_dim=source_te.shape[2], d_ff=256,
                                               heads=2, N=4, out_shape=26, dropout=0.0).to(t.device)

    train_ds_tr_N = TensorDataset(source_tr_N[:, 0:-1], source_tr_N[:, 1:])
    train_dl_tr_N = DataLoader(train_ds_tr_N, bs, shuffle=True)
    train_ds_va = TensorDataset(source_va[:, 0:-1], source_va[:, 1:])
    train_dl_va = DataLoader(train_ds_va, bs, shuffle=False)
    train_ds_te = TensorDataset(source_te[:, 0:-1], source_te[:, 1:])
    train_dl_te = DataLoader(train_ds_te, bs, shuffle=False)

    cost_N = torch.nn.MSELoss()
    cost_MAE = torch.nn.L1Loss()

    t.model_load_eval(N_transformer_model, path="./model_save/N/t_N_va.pth")
    N_transformer_model.eval()

    batch_loss_tr_N = []
    batch_loss_va = []
    batch_loss_te = []
    for x_tr_N, y_tr_N in train_dl_tr_N:
        out_tr_N = N_transformer_model(x_tr_N, mask)
        loss_tr_N = cost_N(out_tr_N, y_tr_N)
        batch_loss_tr_N.append(loss_tr_N.detach().cpu().numpy() * x_tr_N.shape[0])
    loss_tr_N_all = np.sum(batch_loss_tr_N) / source_tr_N.shape[0]

    for x_va, y_va in train_dl_va:
        out_va = N_transformer_model(x_va, mask)
        loss_va = cost_N(out_va, y_va)
        batch_loss_va.append(loss_va.detach().cpu().numpy() * x_va.shape[0])
    loss_va_all = np.sum(batch_loss_va) / source_va.shape[0]

    for x_te, y_te in train_dl_te:
        out_te = N_transformer_model(x_te, mask)
        loss_te = cost_N(out_te, y_te)
        batch_loss_te.append(loss_te.detach().cpu().numpy() * x_te.shape[0])
        print('MAE =', '{:.9f}'.format(cost_MAE(out_te, y_te)))
        print('MSE =', '{:.9f}'.format(cost_N(out_te, y_te)))
        print('RMSE=', '{:.9f}'.format(torch.sqrt(cost_N(out_te, y_te))))
    loss_te_all = np.sum(batch_loss_te) / source_te.shape[0]

    print("-----------------------------")
    print('loss_tr_N=', '{:.9f}'.format(loss_tr_N_all))
    print('loss_va_N=', '{:.9f}'.format(loss_va_all))
    print('loss_te_N=', '{:.9f}'.format(loss_te_all))
