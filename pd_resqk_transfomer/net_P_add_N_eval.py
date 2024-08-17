import glob

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import net_model.pd_transformer_net as t
import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


def model_load(net, path="./model_save/t.pth"):
    net.load_state_dict(torch.load(path))
    print("读取模型：" + path)


def get_data_dl(source, target, net, mask, shuffle=False):
    net.eval()
    train_ds = TensorDataset(source, target)
    train_dl = DataLoader(train_ds, 1000, shuffle=False)
    p_x = torch.empty(0, 71, 26).to(t.device)
    for x, _ in train_dl:
        net_y = net(x, mask).data
        p_x = torch.cat([p_x, net_y], dim=0)
    train_ds = TensorDataset(source, p_x, target)
    train_dl_out = DataLoader(train_ds, bs, shuffle=shuffle)

    return train_dl_out


if __name__ == '__main__':
    bs = 1000
    features_tr_N, _ = t.get_data_("./data/datasets/train_set_people.npy", NP="N",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_tr_P, _ = t.get_data_("./data/datasets/train_set_people.npy", NP="P",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_va_N, _ = t.get_data_("./data/datasets/va_set_people.npy", NP="N",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_va_P, _ = t.get_data_("./data/datasets/va_set_people.npy", NP="P",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_te_N, _ = t.get_data_("./data/datasets/test_set_people.npy", NP="N",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_te_P, _ = t.get_data_("./data/datasets/test_set_people.npy", NP="P",
                                   sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    source_tr_N = features_tr_N.to(t.device)
    source_tr_P = features_tr_P.to(t.device)
    source_va_N = features_va_N.to(t.device)
    source_va_P = features_va_P.to(t.device)
    source_te_N = features_te_N.to(device)
    source_te_P = features_te_P.to(device)

    mask = t.shape_mask(source_te_N.shape[1] - 1, source_te_N.shape[1] - 1, k=1).to(device)
    N_transformer_model = t.MakeTransformer(max_len=source_te_N.shape[1], e_dim=source_te_N.shape[2], d_ff=256,
                                            heads=2, N=4, out_shape=26, dropout=0.0).to(device)
    model_load(N_transformer_model, path="./model_save/N/t_N_va.pth")
    train_dl_tr_N = get_data_dl(source_tr_N[:, 0:-1], source_tr_N[:, 1:], N_transformer_model, mask, shuffle=False)
    train_dl_tr_P = get_data_dl(source_tr_P[:, 0:-1], source_tr_P[:, 1:], N_transformer_model, mask, shuffle=False)
    train_dl_va_N = get_data_dl(source_va_N[:, 0:-1], source_va_N[:, 1:], N_transformer_model, mask, shuffle=False)
    train_dl_va_P = get_data_dl(source_va_P[:, 0:-1], source_va_P[:, 1:], N_transformer_model, mask, shuffle=False)
    train_dl_te_N = get_data_dl(source_te_N[:, 0:-1], source_te_N[:, 1:], N_transformer_model, mask, shuffle=False)
    train_dl_te_P = get_data_dl(source_te_P[:, 0:-1], source_te_P[:, 1:], N_transformer_model, mask, shuffle=False)

    P_transformer_model = t.MakeTransformer(max_len=source_te_P.shape[1], e_dim=source_te_P.shape[2] * 2, d_ff=256,
                                            heads=4, N=6, out_shape=26, dropout=0.0).to(device)

    cost_NP = torch.nn.MSELoss()

    model_load(P_transformer_model, path="./model_save/FN/t_P_va.pth")
    P_transformer_model.eval()

    batch_loss_tr_NN = []
    batch_loss_tr_NA = []
    batch_loss_tr_NZ = []
    batch_loss_tr_PN = []
    batch_loss_tr_PA = []
    batch_loss_tr_PZ = []

    for x_tr_N, x_tr_NN, y_tr_N in train_dl_tr_N:
        out_tr_N1 = x_tr_NN
        out_tr_N2 = P_transformer_model(torch.cat([x_tr_N, x_tr_NN], dim=2), mask)
        out_tr_NA = out_tr_N1 + out_tr_N2

        loss_tr_NN = cost_NP(out_tr_N1, y_tr_N)
        loss_tr_NA = cost_NP(out_tr_NA, y_tr_N)
        loss_tr_NZ = cost_NP(out_tr_N2, torch.zeros(out_tr_N2.shape).to(t.device))

        batch_loss_tr_NN.append(loss_tr_NN.detach().cpu().numpy() * x_tr_N.shape[0])
        batch_loss_tr_NA.append(loss_tr_NA.detach().cpu().numpy() * x_tr_N.shape[0])
        batch_loss_tr_NZ.append(loss_tr_NZ.detach().cpu().numpy() * x_tr_N.shape[0])
    loss_tr_NN_all = np.sum(batch_loss_tr_NN) / source_tr_N.shape[0]
    loss_tr_NA_all = np.sum(batch_loss_tr_NA) / source_tr_N.shape[0]
    loss_tr_NZ_all = np.sum(batch_loss_tr_NZ) / source_tr_N.shape[0]

    for x_tr_P, x_tr_PN, y_tr_P in train_dl_tr_P:
        out_tr_P1 = x_tr_PN
        out_tr_P2 = P_transformer_model(torch.cat([x_tr_P, x_tr_PN], dim=2), mask)
        out_tr_PA = out_tr_P1 + out_tr_P2

        loss_tr_PN = cost_NP(out_tr_P1, y_tr_P)
        loss_tr_PA = cost_NP(out_tr_PA, y_tr_P)
        loss_tr_PZ = cost_NP(out_tr_P2, torch.zeros(out_tr_P2.shape).to(t.device))

        batch_loss_tr_PN.append(loss_tr_PN.detach().cpu().numpy() * x_tr_P.shape[0])
        batch_loss_tr_PA.append(loss_tr_PA.detach().cpu().numpy() * x_tr_P.shape[0])
        batch_loss_tr_PZ.append(loss_tr_PZ.detach().cpu().numpy() * x_tr_P.shape[0])
    loss_tr_PN_all = np.sum(batch_loss_tr_PN) / source_tr_P.shape[0]
    loss_tr_PA_all = np.sum(batch_loss_tr_PA) / source_tr_P.shape[0]
    loss_tr_PZ_all = np.sum(batch_loss_tr_PZ) / source_tr_P.shape[0]

    loss_tr_N_all = (np.sum(batch_loss_tr_NN) + np.sum(batch_loss_tr_PN)) / (
            source_tr_N.shape[0] + source_tr_P.shape[0])
    loss_tr_A_all = (np.sum(batch_loss_tr_NA) + np.sum(batch_loss_tr_PA)) / (
            source_tr_N.shape[0] + source_tr_P.shape[0])
    loss_tr_Z_all = (np.sum(batch_loss_tr_NZ) + np.sum(batch_loss_tr_PZ)) / (
            source_tr_N.shape[0] + source_tr_P.shape[0])

    batch_loss_va_NN = []
    batch_loss_va_NA = []
    batch_loss_va_NZ = []
    batch_loss_va_PN = []
    batch_loss_va_PA = []
    batch_loss_va_PZ = []

    for x_va_N, x_va_NN, y_va_N in train_dl_va_N:
        out_va_N1 = x_va_NN
        out_va_N2 = P_transformer_model(torch.cat([x_va_N, x_va_NN], dim=2), mask)
        out_va_NA = out_va_N1 + out_va_N2

        loss_va_NN = cost_NP(out_va_N1, y_va_N)
        loss_va_NA = cost_NP(out_va_NA, y_va_N)
        loss_va_NZ = cost_NP(out_va_N2, torch.zeros(out_va_N2.shape).to(t.device))

        batch_loss_va_NN.append(loss_va_NN.detach().cpu().numpy() * x_va_N.shape[0])
        batch_loss_va_NA.append(loss_va_NA.detach().cpu().numpy() * x_va_N.shape[0])
        batch_loss_va_NZ.append(loss_va_NZ.detach().cpu().numpy() * x_va_N.shape[0])
    loss_va_NN_all = np.sum(batch_loss_va_NN) / source_va_N.shape[0]
    loss_va_NA_all = np.sum(batch_loss_va_NA) / source_va_N.shape[0]
    loss_va_NZ_all = np.sum(batch_loss_va_NZ) / source_va_N.shape[0]

    for x_va_P, x_va_PN, y_va_P in train_dl_va_P:
        out_va_P1 = x_va_PN
        out_va_P2 = P_transformer_model(torch.cat([x_va_P, x_va_PN], dim=2), mask)
        out_va_PA = out_va_P1 + out_va_P2

        loss_va_PN = cost_NP(out_va_P1, y_va_P)
        loss_va_PA = cost_NP(out_va_PA, y_va_P)
        loss_va_PZ = cost_NP(out_va_P2, torch.zeros(out_va_P2.shape).to(t.device))

        batch_loss_va_PN.append(loss_va_PN.detach().cpu().numpy() * x_va_P.shape[0])
        batch_loss_va_PA.append(loss_va_PA.detach().cpu().numpy() * x_va_P.shape[0])
        batch_loss_va_PZ.append(loss_va_PZ.detach().cpu().numpy() * x_va_P.shape[0])
    loss_va_PN_all = np.sum(batch_loss_va_PN) / source_va_P.shape[0]
    loss_va_PA_all = np.sum(batch_loss_va_PA) / source_va_P.shape[0]
    loss_va_PZ_all = np.sum(batch_loss_va_PZ) / source_va_P.shape[0]

    loss_va_N_all = (np.sum(batch_loss_va_NN) + np.sum(batch_loss_va_PN)) / (
            source_va_N.shape[0] + source_va_P.shape[0])
    loss_va_A_all = (np.sum(batch_loss_va_NA) + np.sum(batch_loss_va_PA)) / (
            source_va_N.shape[0] + source_va_P.shape[0])
    loss_va_Z_all = (np.sum(batch_loss_va_NZ) + np.sum(batch_loss_va_PZ)) / (
            source_va_N.shape[0] + source_va_P.shape[0])

    batch_loss_te_NN = []
    batch_loss_te_NA = []
    batch_loss_te_NZ = []
    batch_loss_te_PN = []
    batch_loss_te_PA = []
    batch_loss_te_PZ = []

    yp_te_NN = torch.empty(0, source_te_N[:, 0:-1].shape[1], source_te_N[:, 0:-1].shape[2])
    yp_te_NA = torch.empty(0, source_te_N[:, 0:-1].shape[1], source_te_N[:, 0:-1].shape[2])

    yp_te_PN = torch.empty(0, source_te_N[:, 0:-1].shape[1], source_te_N[:, 0:-1].shape[2])
    yp_te_PA = torch.empty(0, source_te_N[:, 0:-1].shape[1], source_te_N[:, 0:-1].shape[2])

    for x_te_N, x_te_NN, y_te_N in train_dl_te_N:
        out_te_N1 = x_te_NN
        out_te_N2 = P_transformer_model(torch.cat([x_te_N, x_te_NN], dim=2), mask)
        out_te_NA = out_te_N1 + out_te_N2

        loss_te_NN = cost_NP(out_te_N1, y_te_N)
        loss_te_NA = cost_NP(out_te_NA, y_te_N)
        loss_te_NZ = cost_NP(out_te_N2, torch.zeros(out_te_N2.shape).to(t.device))
        print('{:.9f}'.format(loss_te_NZ))

        yp_te_NN = torch.cat([yp_te_NN, out_te_N1.cpu()], dim=0)
        yp_te_NA = torch.cat([yp_te_NA, out_te_NA.cpu()], dim=0)

        batch_loss_te_NN.append(loss_te_NN.detach().cpu().numpy() * x_te_N.shape[0])
        batch_loss_te_NA.append(loss_te_NA.detach().cpu().numpy() * x_te_N.shape[0])
        batch_loss_te_NZ.append(loss_te_NZ.detach().cpu().numpy() * x_te_N.shape[0])
    loss_te_NN_all = np.sum(batch_loss_te_NN) / source_te_N.shape[0]
    loss_te_NA_all = np.sum(batch_loss_te_NA) / source_te_N.shape[0]
    loss_te_NZ_all = np.sum(batch_loss_te_NZ) / source_te_N.shape[0]

    for x_te_P, x_te_PN, y_te_P in train_dl_te_P:
        out_te_P1 = x_te_PN
        out_te_P2 = P_transformer_model(torch.cat([x_te_P, x_te_PN], dim=2), mask)
        out_te_PA = out_te_P1 + out_te_P2

        loss_te_PN = cost_NP(out_te_P1, y_te_P)
        loss_te_PA = cost_NP(out_te_PA, y_te_P)
        loss_te_PZ = cost_NP(out_te_P2, torch.zeros(out_te_P2.shape).to(t.device))

        yp_te_PN = torch.cat([yp_te_PN, out_te_P1.cpu()], dim=0)
        yp_te_PA = torch.cat([yp_te_PA, out_te_PA.cpu()], dim=0)

        batch_loss_te_PN.append(loss_te_PN.detach().cpu().numpy() * x_te_P.shape[0])
        batch_loss_te_PA.append(loss_te_PA.detach().cpu().numpy() * x_te_P.shape[0])
        batch_loss_te_PZ.append(loss_te_PZ.detach().cpu().numpy() * x_te_P.shape[0])
    loss_te_PN_all = np.sum(batch_loss_te_PN) / source_te_P.shape[0]
    loss_te_PA_all = np.sum(batch_loss_te_PA) / source_te_P.shape[0]
    loss_te_PZ_all = np.sum(batch_loss_te_PZ) / source_te_P.shape[0]

    loss_te_N_all = (np.sum(batch_loss_te_NN) + np.sum(batch_loss_te_PN)) / (
            source_te_N.shape[0] + source_te_P.shape[0])
    loss_te_A_all = (np.sum(batch_loss_te_NA) + np.sum(batch_loss_te_PA)) / (
            source_te_N.shape[0] + source_te_P.shape[0])
    loss_te_Z_all = (np.sum(batch_loss_te_NZ) + np.sum(batch_loss_te_PZ)) / (
            source_te_N.shape[0] + source_te_P.shape[0])

    print("tr--------------------------------------------")
    print("loss_tr_NN:", '{:.9f}'.format(loss_tr_NN_all), "loss_tr_NA:", '{:.9f}'.format(loss_tr_NA_all), "loss_tr_NZ:",
          '{:.9f}'.format(loss_tr_NZ_all))
    print("loss_tr_PN:", '{:.9f}'.format(loss_tr_PN_all), "loss_tr_PA:", '{:.9f}'.format(loss_tr_PA_all), "loss_tr_PZ:",
          '{:.9f}'.format(loss_tr_PZ_all))
    print("loss_tr_N :", '{:.9f}'.format(loss_tr_N_all), "loss_tr_A :", '{:.9f}'.format(loss_tr_A_all), "loss_tr_Z :",
          '{:.9f}'.format(loss_tr_Z_all))
    print("va--------------------------------------------")
    print("loss_va_NN:", '{:.9f}'.format(loss_va_NN_all), "loss_va_NA:", '{:.9f}'.format(loss_va_NA_all), "loss_va_NZ:",
          '{:.9f}'.format(loss_va_NZ_all))
    print("loss_va_PN:", '{:.9f}'.format(loss_va_PN_all), "loss_va_PA:", '{:.9f}'.format(loss_va_PA_all), "loss_va_PZ:",
          '{:.9f}'.format(loss_va_PZ_all))
    print("loss_va_N :", '{:.9f}'.format(loss_va_N_all), "loss_va_A :", '{:.9f}'.format(loss_va_A_all), "loss_va_Z :",
          '{:.9f}'.format(loss_va_Z_all))

    # # 患者步态
    # yp_te_PN = yp_te_PN.view(-1, 71, 2, 13).detach().cpu().numpy()
    # yp_te_PA = yp_te_PA.view(-1, 71, 2, 13).detach().cpu().numpy()
    # yt_te = source_te_P[:, 1:].view(-1, 71, 2, 13).detach().cpu().numpy()
    #
    # fig, ax = plt.subplots()
    # for j in range(300, 301):
    #     pics_list = []
    #     for i in range(71):
    #         ax.clear()
    #         ax.plot(yt_te[j][i, 0, :], yt_te[j][i, 1, :], 'o', color="b", markersize=4)
    #
    #         ax.set_xlim([-0.6, 0.6])
    #         ax.set_ylim([-0.6, 0.6])
    #         plt.axis('off')
    #         png = plt.savefig("./png_save/temp/" + str(j) + "_" + str(i) + "_t.png")
    #         plt.savefig("./png_save/temp/" + str(j) + "_" + str(i) + "_t.eps")
    #         plt.pause(1 / 24)
    #
    # plt.close()


