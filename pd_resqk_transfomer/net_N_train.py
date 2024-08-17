import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import net_model.pd_transformer_net as t

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    lr = 0.00001
    bs = 8000
    ep = 10000000

    features_tr_N, _ = t.get_data_("./data/datasets/train_set_people.npy", NP="N",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_va, _ = t.get_data_("./data/datasets/va_set_people.npy", NP="N",
                                 sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    source_tr_N = features_tr_N.to(t.device)
    source_va = features_va.to(t.device)

    print(source_tr_N.shape[1], source_tr_N.shape[2])

    mask = t.shape_mask(source_tr_N.shape[1] - 1, source_tr_N.shape[1] - 1, k=1).to(t.device)
    N_transformer_model = t.MakeTransformerOri(max_len=source_tr_N.shape[1], e_dim=source_tr_N.shape[2], d_ff=256,
                                               heads=2, N=4, out_shape=26, dropout=0.0).to(t.device)


    train_ds_tr_N = TensorDataset(source_tr_N[:, 0:-1], source_tr_N[:, 1:])
    train_dl_tr_N = DataLoader(train_ds_tr_N, bs, shuffle=True)
    train_ds_va = TensorDataset(source_va[:, 0:-1], source_va[:, 1:])
    train_dl_va = DataLoader(train_ds_va, bs, shuffle=False)

    cost_N = torch.nn.MSELoss()
    # cost_N = torch.nn.SmoothL1Loss()
    cost_VA = torch.nn.MSELoss()

    op_N = torch.optim.RAdam(N_transformer_model.parameters(), betas=(0.9, 0.99), lr=lr, weight_decay=1e-27)


    for epoch in range(ep):
        batch_loss_tr_N = []
        batch_loss_va = []
        batch_loss_te = []
        for x_tr_N, y_tr_N in train_dl_tr_N:
            N_transformer_model.train()
            out_tr_N = N_transformer_model(x_tr_N, mask)
            loss_tr_N = cost_N(out_tr_N, y_tr_N)
            op_N.zero_grad()
            loss_tr_N.backward()
            op_N.step()
            batch_loss_tr_N.append(loss_tr_N.detach().cpu().numpy() * x_tr_N.shape[0])
        loss_tr_N_all = np.sum(batch_loss_tr_N) / source_tr_N.shape[0]

        for x_va, y_va in train_dl_va:
            N_transformer_model.eval()
            out_va = N_transformer_model(x_va, mask)
            loss_va = cost_VA(out_va, y_va)
            batch_loss_va.append(loss_va.detach().cpu().numpy() * x_va.shape[0])
        loss_va_all = np.sum(batch_loss_va) / source_va.shape[0]

        print("-----------------------------")
        print('Epoch:', '%04d' % (epoch + 1))
        print('loss_tr_N=', '{:.9f}'.format(loss_tr_N_all))
        print('loss_va_N=', '{:.9f}'.format(loss_va_all))

        t.model_save(loss_tr_N_all, "loss", N_transformer_model, max_min="min",
                     argument_path="model_save/N/t_N_tr.csv", net_path="model_save/N/t_N_tr.pth")
        t.model_save(loss_va_all, "loss", N_transformer_model, max_min="min",
                     argument_path="model_save/N/t_N_va.csv", net_path="model_save/N/t_N_va.pth")
