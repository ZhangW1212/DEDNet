import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset, DailyDialogDataset
from model import MaskedNLLLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import random
from vision import TwoD_Tsne, sampleCount, confuPLT

seed = 2094
print('seed = {}'.format(seed))

def seed_everything():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything()

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, windows=5):
    trainset = MELDDataset('data/meld_multimodal_features.pkl', windows=windows)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False, windows=windows)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             # 随机样本可视化注意力系数时使用
                             # shuffle=True,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=True, windows=5):
    trainset = IEMOCAPDataset("data/iemocap_multimodal_features.pkl", windows=windows)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset("data/iemocap_multimodal_features.pkl", train=False, windows=windows)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             # 随机样本可视化注意力系数时使用
                             # shuffle=True,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

def get_DailyDialog_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=True, windows=5):
    trainset = DailyDialogDataset("data/dd_multimodal_features.pkl", windows=windows)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = DailyDialogDataset("data/dd_multimodal_features.pkl", train=False, windows=windows)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             # 随机样本可视化注意力系数时使用
                             # shuffle=True,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader

# model, loss_function, kl_loss, train_loader, e, optimizer, True
def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False, dataset=None):
    losses, preds, labels, masks, all_transformer_outs, umasks = [], [], [], [], [], []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        if  dataset=="DailyDialog":
            textf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj\
                    = [d.cuda() for d in data] if cuda else data
            visuf = None
            acouf = None
        else:
            textf, visuf, acouf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj\
                = [d.cuda() for d in data] if cuda else data

        umasks.append(umask)
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        log_prob, all_log_prob, all_prob, all_transformer_out = \
            model(textf, visuf, acouf, umask, qmask, lengths, Self_semantic_adj, Cross_semantic_adj, Semantic_adj)
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        #融合损失
        fusion_loss = loss_function(lp_all, labels_, umask)
        if dataset=="DailyDialog":
            loss = fusion_loss
        else:
            # 单模态损失
            t_loss = loss_function(log_prob[0].view(-1, log_prob[0].size()[2]), labels_, umask)
            a_loss = loss_function(log_prob[1].view(-1, log_prob[1].size()[2]), labels_, umask)
            v_loss = loss_function(log_prob[2].view(-1, log_prob[2].size()[2]), labels_, umask)
            #总损失
            loss = fusion_loss + t_loss * (t_loss / 10) + a_loss * (a_loss / 10) + v_loss * (v_loss / 10)

        lp_ = all_prob.view(-1, all_prob.size()[2])
        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        all_transformer_outs.append(all_transformer_out)
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()
    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), [], []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, all_transformer_outs, umasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--l2', type=float, default=0.00005, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=512, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--windows', type=int, default=20, help='number of windows')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='DailyDialog', help='dataset to train and test')
    parser.add_argument('--save_model_path', default='./IEMOCAP', type=str, help='模型输出路径')
    args = parser.parse_args()
    print(args)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10':1582, 'denseface':342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 6 if args.Dataset=='IEMOCAP' else 7

    model = Transformer_Based_Model(args.Dataset, D_text, D_visual, D_audio, args.n_head,
                                                        n_classes=n_classes,
                                                        hidden_dim=args.hidden_dim,
                                                        n_speakers=n_speakers,
                                                        dropout=args.dropout)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))
    if torch.cuda.device_count()>0:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    if args.Dataset == 'MELD':
        loss_weights = torch.FloatTensor([1/0.481226,
                                          1/0.107663,
                                          1/0.191571,
                                          1/0.079669,
                                          1/0.154023,
                                          1/0.026054,
                                          1/0.132184
                                            ])
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.1,
                                                                     batch_size=batch_size,
                                                                    num_workers=0, windows=args.windows)
    elif args.Dataset == 'IEMOCAP':
        # 这个权重是每一类的样本数与总样本数的比例的倒数，这样定义loss的作用是对样本不均衡的类别给予更高的权重
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.cuda())
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.1,
                                                                      batch_size=batch_size,
                                                                      num_workers=0, windows=args.windows)

    else:
        loss_weights = torch.FloatTensor([1 / 0.0017,
                                      1 / 0.0034,
                                      1 / 0.1251,
                                      1 / 0.831,
                                      1 / 0.0099,
                                      1 / 0.0177,
                                      1 / 0.0112])
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_DailyDialog_loaders(valid=0.1,
                                                                      batch_size=batch_size,
                                                                      num_workers=0, windows=args.windows)

    best_fscore, best_loss, best_label, best_pred, best_mask, best_feature, best_umasks= None, None, None, None, None, None, None
    all_valid_fscore, all_fscore, all_acc, all_loss = [], [], [], []
    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _, _ = train_or_eval_model(model, loss_function, train_loader, optimizer, train=True, dataset=args.Dataset)
        valid_loss, valid_acc, _, _, _, valid_fscore, _, _ = train_or_eval_model(model, loss_function, valid_loader, train=False, dataset=args.Dataset)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, all_transformer_outs, umasks = train_or_eval_model(model, loss_function, test_loader, train=False, dataset=args.Dataset)
        all_valid_fscore.append(test_fscore)
        all_fscore.append(test_fscore)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask, best_feature = test_label, test_pred, test_mask, all_transformer_outs
        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))

    save_path = os.path.join("./"+args.Dataset, "bestModel.pth")
    torch.save(model, save_path)
    print('Model Performance:')
    print('Best_Test-FScore-epoch_index: {}'.format(all_fscore.index(max(all_fscore))+1))
    print('Best_Test_F-Score: {}'.format(max(all_fscore)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
    confuPLT(confusion_matrix(best_label, best_pred, sample_weight=best_mask).astype(int), args.Dataset)


