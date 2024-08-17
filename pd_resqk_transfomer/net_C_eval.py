import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import net_model.pd_transformer_net as t

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_load(net, path="./model_save/t.pth"):
    net.load_state_dict(torch.load(path))
    print("读取模型：" + path)
    return


def get_data_dl0(source, target, net, mask, bs=1000, shuffle=False):
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


def get_data_dl1(features, label, bs=1000, shuffle=False):
    source = features.to(t.device)

    mask = t.shape_mask(source.shape[1] - 1, source.shape[1] - 1, k=1).to(t.device)
    N_transformer_model = t.MakeTransformer(max_len=source.shape[1], e_dim=source.shape[2], d_ff=256,
                                            heads=2, N=4, out_shape=26, dropout=0.0).to(t.device)
    model_load(N_transformer_model, path="./model_save/N/t_N_va.pth")

    train_dl = get_data_dl0(source[:, 0:-1], source[:, 1:], N_transformer_model, mask, bs=bs, shuffle=False)

    P_transformer_model = t.MakeTransformer(max_len=source.shape[1], e_dim=source.shape[2] * 2, d_ff=256,
                                            heads=4, N=6, out_shape=26, dropout=0.0).to(t.device)
    model_load(P_transformer_model, path="./model_save/FN/t_P_va.pth")

    P_transformer_model.eval()
    p_x = torch.empty(0, 71, 26).to(t.device)
    for x, x_N, _ in train_dl:
        out = P_transformer_model(torch.cat([x, x_N], dim=2), mask).data
        p_x = torch.cat([p_x, out], dim=0)

    x_out = p_x.view(p_x.shape[0], 71, 2, 13)
    source = source.view(source.shape[0], 72, 2, 13)
    y = label.to(t.device)
    train_ds = TensorDataset(x_out, source, y)
    train_dl_out = DataLoader(train_ds, bs, shuffle=shuffle)

    return train_dl_out


def output_index(y_p, y_t):
    inputs = y_p.detach().cpu().numpy().argmax(axis=1)
    labels = y_t.detach().cpu().numpy().argmax(axis=1)
    TP = sum([1 for i in range(len(inputs)) if inputs[i] == labels[i] and inputs[i] == 1])
    FP = sum([1 for i in range(len(inputs)) if inputs[i] != labels[i] and inputs[i] == 1])
    TN = sum([1 for i in range(len(inputs)) if inputs[i] == labels[i] and inputs[i] == 0])
    FN = sum([1 for i in range(len(inputs)) if inputs[i] != labels[i] and inputs[i] == 0])
    acc = (TP + TN + 0.00001) / (TP + TN + FP + FN + 0.00001)
    precision = (TP + 0.00001) / (TP + FP + 0.00001)
    recall = (TP + 0.00001) / (TP + FN + 0.00001)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('准确率=', '{:.9f}'.format(acc), 'F1分数=', '{:.9f}'.format(f1_score))
    print('精确率=', '{:.9f}'.format(precision), '召回率=', '{:.9f}'.format(recall))

    return acc, precision, recall, f1_score


def eval_one_batch(net, label, *args):
    net.eval()
    out = net(*args)
    loss = cost_C(out, label)
    return out, loss


if __name__ == '__main__':

    features_tr, label_tr = t.get_data_("./data/datasets/train_set_people.npy", NP="A",
                                        sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_va, label_va = t.get_data_("./data/datasets/va_set_people.npy", NP="A",
                                        sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    features_te, label_te = t.get_data_("./data/datasets/test_set_people.npy", NP="A",
                                        sk_id=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

    classification_71_net = t.ClassificationNet71A().to(t.device)
    model_load(classification_71_net, path="./model_save/C/C_net_va_71.pth")

    classification_72_net = t.ClassificationNet72A().to(t.device)
    model_load(classification_72_net, path="./model_save/C/C_net_va_72.pth")

    classification_add_net = t.ClassificationNetAddA().to(t.device)
    model_load(classification_add_net, path="./model_save/C/C_net_va_add.pth")

    cost_C = torch.nn.CrossEntropyLoss()

    train_dl_tr = get_data_dl1(features_tr, label_tr, bs=1000, shuffle=False)
    train_dl_va = get_data_dl1(features_va, label_va, bs=1000, shuffle=False)
    train_dl_te = get_data_dl1(features_te, label_te, bs=1000, shuffle=False)

    batch_loss_71_C = []
    batch_loss_71_va = []
    batch_loss_71_te = []
    batch_loss_72_C = []
    batch_loss_72_va = []
    batch_loss_72_te = []
    batch_loss_add_C = []
    batch_loss_add_va = []
    batch_loss_add_te = []
    tr_71_out = torch.empty(0, 2).to(t.device)
    va_71_out = torch.empty(0, 2).to(t.device)
    te_71_out = torch.empty(0, 2).to(t.device)
    tr_72_out = torch.empty(0, 2).to(t.device)
    va_72_out = torch.empty(0, 2).to(t.device)
    te_72_out = torch.empty(0, 2).to(t.device)
    tr_add_out = torch.empty(0, 2).to(t.device)
    va_add_out = torch.empty(0, 2).to(t.device)
    te_add_out = torch.empty(0, 2).to(t.device)
    tr_label = torch.empty(0, 2).to(t.device)
    for x_C, x_C_, y_C in train_dl_tr:
        out_71_C, loss_71_C = eval_one_batch(classification_71_net, y_C, x_C)
        out_72_C, loss_72_C = eval_one_batch(classification_72_net, y_C, x_C_)
        out_add_C, loss_add_C = eval_one_batch(classification_add_net, y_C, x_C, x_C_)

        tr_71_out = torch.cat([tr_71_out, out_71_C], dim=0)
        tr_72_out = torch.cat([tr_72_out, out_72_C], dim=0)
        tr_add_out = torch.cat([tr_add_out, out_add_C], dim=0)
        tr_label = torch.cat([tr_label, y_C], dim=0)

        batch_loss_71_C.append(loss_71_C.detach().cpu().numpy() * x_C.shape[0])
        batch_loss_72_C.append(loss_72_C.detach().cpu().numpy() * x_C.shape[0])
        batch_loss_add_C.append(loss_add_C.detach().cpu().numpy() * x_C.shape[0])

    loss_71_C_all = np.sum(batch_loss_71_C) / features_tr.shape[0]
    loss_72_C_all = np.sum(batch_loss_72_C) / features_tr.shape[0]
    loss_add_C_all = np.sum(batch_loss_add_C) / features_tr.shape[0]

    for x_va, x_va_, y_va in train_dl_va:
        out_71_va, loss_71_va = eval_one_batch(classification_71_net, y_va, x_va)
        out_72_va, loss_72_va = eval_one_batch(classification_72_net, y_va, x_va_)
        out_add_va, loss_add_va = eval_one_batch(classification_add_net, y_va, x_va, x_va_)

        va_71_out = torch.cat([va_71_out, out_71_va], dim=0)
        va_72_out = torch.cat([va_72_out, out_72_va], dim=0)
        va_add_out = torch.cat([va_add_out, out_add_va], dim=0)

        batch_loss_71_va.append(loss_71_va.detach().cpu().numpy() * x_va.shape[0])
        batch_loss_72_va.append(loss_72_va.detach().cpu().numpy() * x_va.shape[0])
        batch_loss_add_va.append(loss_add_va.detach().cpu().numpy() * x_va.shape[0])
    loss_71_va_all = np.sum(batch_loss_71_va) / features_va.shape[0]
    loss_72_va_all = np.sum(batch_loss_72_va) / features_va.shape[0]
    loss_add_va_all = np.sum(batch_loss_add_va) / features_va.shape[0]

    for x_te, x_te_, y_te in train_dl_te:
        out_71_te, loss_71_te = eval_one_batch(classification_71_net, y_te, x_te)
        out_72_te, loss_72_te = eval_one_batch(classification_72_net, y_te, x_te_)
        out_add_te, loss_add_te = eval_one_batch(classification_add_net, y_te, x_te, x_te_)

        te_71_out = torch.cat([te_71_out, out_71_te], dim=0)
        te_72_out = torch.cat([te_72_out, out_72_te], dim=0)
        te_add_out = torch.cat([te_add_out, out_add_te], dim=0)

        batch_loss_71_te.append(loss_71_te.detach().cpu().numpy() * x_te.shape[0])
        batch_loss_72_te.append(loss_72_te.detach().cpu().numpy() * x_te.shape[0])
        batch_loss_add_te.append(loss_add_te.detach().cpu().numpy() * x_te.shape[0])
    loss_71_te_all = np.sum(batch_loss_71_te) / features_te.shape[0]
    loss_72_te_all = np.sum(batch_loss_72_te) / features_te.shape[0]
    loss_add_te_all = np.sum(batch_loss_add_te) / features_te.shape[0]

    print("71-----------------------------")
    print('loss_71_tr_N=', '{:.9f}'.format(loss_71_C_all))
    print('loss_71_va_N=', '{:.9f}'.format(loss_71_va_all))
    print('loss_71_te_N=', '{:.9f}'.format(loss_71_te_all))
    print("训练集指标")
    output_index(tr_71_out, tr_label)
    print("验证集指标")
    output_index(va_71_out, label_va)
    print("测试集指标")
    output_index(te_71_out, label_te)
    print("72-----------------------------")
    print('loss_72_tr_N=', '{:.9f}'.format(loss_72_C_all))
    print('loss_72_va_N=', '{:.9f}'.format(loss_72_va_all))
    print('loss_72_te_N=', '{:.9f}'.format(loss_72_te_all))
    print("训练集指标")
    output_index(tr_72_out, tr_label)
    print("验证集指标")
    output_index(va_72_out, label_va)
    print("测试集指标")
    output_index(te_72_out, label_te)
    print("add-----------------------------")
    print('loss_add_tr_N=', '{:.9f}'.format(loss_add_C_all))
    print('loss_add_va_N=', '{:.9f}'.format(loss_add_va_all))
    print('loss_add_te_N=', '{:.9f}'.format(loss_add_te_all))
    print("训练集指标")
    output_index(tr_add_out, tr_label)
    print("验证集指标")
    output_index(va_add_out, label_va)
    print("测试集指标")
    output_index(te_add_out, label_te)
