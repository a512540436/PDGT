import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import net_model.pd_transformer_net as t

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data_dl(source, target, net, mask, shuffle=False):
    net.eval()
    train_ds = TensorDataset(source, target)
    train_dl = DataLoader(train_ds, bs, shuffle=False)
    p_x = torch.empty(0, 71, 26).to(t.device)
    for x, _ in train_dl:
        net_y = net(x, mask).data
        p_x = torch.cat([p_x, net_y], dim=0)
    train_ds = TensorDataset(source, p_x, target)
    train_dl_out = DataLoader(train_ds, bs, shuffle=shuffle)

    return train_dl_out


if __name__ == '__main__':
    lr_P = 0.00001
    lr_Z = 0.00001
    bs = 1000
    ep = 10000000

    features_tr_N, _ = t.get_data_("./data/datasets/train_set_people.npy", NP="N",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_tr_P, _ = t.get_data_("./data/datasets/train_set_people.npy", NP="P",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_tr_A, _ = t.get_data_("./data/datasets/train_set_people.npy", NP="A",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_va, _ = t.get_data_("./data/datasets/va_set_people.npy", NP="A",
                                 sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    source_tr_N = features_tr_N.to(t.device)
    source_tr_P = features_tr_P.to(t.device)
    source_tr_A = features_tr_A.to(t.device)
    source_va_A = features_va.to(t.device)

    mask = t.shape_mask(source_tr_N.shape[1] - 1, source_tr_N.shape[1] - 1, k=1).to(t.device)
    P_transformer_model = t.MakeTransformer(max_len=source_tr_P.shape[1], e_dim=source_tr_P.shape[2] * 2, d_ff=256,
                                            heads=4, N=6, out_shape=26, dropout=0.0).to(t.device)

    N_transformer_model = t.MakeTransformer(max_len=source_tr_N.shape[1], e_dim=source_tr_N.shape[2], d_ff=256,
                                            heads=2, N=4, out_shape=26, dropout=0.0).to(t.device)
    t.model_load_eval(N_transformer_model, path="./model_save/N/t_N_va.pth")
    train_dl_tr_N = get_data_dl(source_tr_N[:, 0:-1], source_tr_N[:, 1:], N_transformer_model, mask, shuffle=True)
    train_dl_tr_P = get_data_dl(source_tr_P[:, 0:-1], source_tr_P[:, 1:], N_transformer_model, mask, shuffle=True)
    train_dl_tr_A = get_data_dl(source_tr_A[:, 0:-1], source_tr_A[:, 1:], N_transformer_model, mask, shuffle=True)
    train_dl_va_A = get_data_dl(source_va_A[:, 0:-1], source_va_A[:, 1:], N_transformer_model, mask, shuffle=False)

    # cost_N = torch.nn.MSELoss()
    cost_NP = torch.nn.SmoothL1Loss()
    cost_VA = torch.nn.MSELoss()

    op_P = torch.optim.RAdam(P_transformer_model.parameters(), betas=(0.9, 0.99), lr=lr_P,
                             weight_decay=1e-27)  # , weight_decay=1e-27
    op_Z = torch.optim.RAdam(P_transformer_model.parameters(), betas=(0.9, 0.99), lr=lr_Z,
                             weight_decay=1e-27)  # , weight_decay=1e-27

    t.model_load_P(P_transformer_model, mask, train_dl_tr_P, cost_NP, lr_P, True, path="./model_save/FN/t_P_va.pth")
    for epoch in range(ep):
        batch_loss_tr_N = []
        batch_loss_tr_NP = []
        batch_loss_tr_NZ = []
        batch_loss_tr_P = []
        batch_loss_tr_PZ = []
        batch_loss_tr_A = []
        batch_loss_va_A = []
        batch_loss_te_A = []
        for x_tr_N, x_tr_NN, y_tr_N in train_dl_tr_N:
            P_transformer_model.train()
            loss_tr_N = cost_NP(x_tr_NN, y_tr_N)
            out_tr_Z = P_transformer_model(torch.cat([x_tr_N, x_tr_NN], dim=2), mask)
            out = x_tr_NN + out_tr_Z
            loss_tr_NP = cost_NP(out, y_tr_N)
            loss_tr_NZ = cost_NP(out_tr_Z, torch.zeros(out_tr_Z.shape).to(t.device))

            batch_loss_tr_N.append(loss_tr_N.detach().cpu().numpy() * x_tr_N.shape[0])
            batch_loss_tr_NP.append(loss_tr_NP.detach().cpu().numpy() * x_tr_N.shape[0])
            batch_loss_tr_NZ.append(loss_tr_NZ.detach().cpu().numpy() * x_tr_N.shape[0])
        loss_tr_N_all = np.sum(batch_loss_tr_N) / source_tr_N.shape[0]
        loss_tr_NP_all = np.sum(batch_loss_tr_NP) / source_tr_N.shape[0]
        loss_tr_NZ_all = np.sum(batch_loss_tr_NZ) / source_tr_N.shape[0]

        for x_tr_P, x_tr_NP, y_tr_P in train_dl_tr_P:
            P_transformer_model.train()
            out_tr_P = P_transformer_model(torch.cat([x_tr_P, x_tr_NP], dim=2), mask)
            out = x_tr_NP + out_tr_P
            loss_tr_P = cost_NP(out, y_tr_P)

            loss_PZ = cost_NP(out_tr_P, torch.zeros(out_tr_P.shape).to(t.device))
            batch_loss_tr_P.append(loss_tr_P.detach().cpu().numpy() * x_tr_P.shape[0])
            batch_loss_tr_PZ.append(loss_PZ.detach().cpu().numpy() * x_tr_P.shape[0])
        loss_tr_P_all = np.sum(batch_loss_tr_P) / source_tr_P.shape[0]
        loss_tr_PZ_all = np.sum(batch_loss_tr_PZ) / source_tr_P.shape[0]

        for x_tr_A, x_tr_NA, y_tr_A in train_dl_tr_A:
            P_transformer_model.train()
            out_tr_A = P_transformer_model(torch.cat([x_tr_A, x_tr_NA], dim=2), mask)
            out = x_tr_NA + out_tr_A
            loss_tr_A = cost_NP(out, y_tr_A)

            op_P.zero_grad()
            loss_tr_A.backward()
            op_P.step()

            batch_loss_tr_A.append(loss_tr_A.detach().cpu().numpy() * x_tr_A.shape[0])
        loss_tr_A_all = np.sum(batch_loss_tr_A) / source_tr_A.shape[0]

        for x_va_A, x_Nva_A, y_va_A in train_dl_va_A:
            P_transformer_model.eval()
            out_va_A = P_transformer_model(torch.cat([x_va_A, x_Nva_A], dim=2), mask)
            out = x_Nva_A + out_va_A
            loss_va_A = cost_VA(out, y_va_A)
            batch_loss_va_A.append(loss_va_A.detach().cpu().numpy() * x_va_A.shape[0])
        loss_va_A_all = np.sum(batch_loss_va_A) / source_va_A.shape[0]

        print("-----------------------------")
        print('Epoch:', '%04d' % (epoch + 1))
        print('loss_tr_N =', '{:.9f}'.format(loss_tr_N_all))
        print('loss_tr_NA=', '{:.9f}'.format(loss_tr_NP_all), 'loss_NZ=', '{:.9f}'.format(loss_tr_NZ_all))
        print('loss_tr_PA=', '{:.9f}'.format(loss_tr_P_all), 'loss_PZ=', '{:.9f}'.format(loss_tr_PZ_all))
        print('loss_tr_A =', '{:.9f}'.format(loss_tr_A_all))
        print('loss_va_A =', '{:.9f}'.format(loss_va_A_all))

        t.model_save(loss_tr_P_all, "loss", P_transformer_model, max_min="min",
                     argument_path="model_save/FN/t_P_tr.csv", net_path="model_save/FN/t_P_tr.pth")
        t.model_save(loss_va_A_all, "loss", P_transformer_model, max_min="min",
                     argument_path="model_save/FN/t_P_va.csv", net_path="model_save/FN/t_P_va.pth")
