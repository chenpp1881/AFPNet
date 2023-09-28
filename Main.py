import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from AFPNet import AFPNet
from tool import all_metrics, save_model
import argparse
from Dataset_process import Dataset_process

parser = argparse.ArgumentParser()
parser.add_argument('--is_train', type=bool, default=True)


# parameters
parser.add_argument('--project', type=str, default='reentrancy',
                    choices=['reentrancy', 'timestamp', 'loops', 'reentrancy_cleaned', 'timestamp_cleaned'])
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--resume', type=str)
parser.add_argument('--save_model_path', type=str, default='./Result')
parser.add_argument('--seq_model', type=str, default='Att', choices=['Att', 'RNN', 'LSTM', 'RNN'])
parser.add_argument('--return_indecs',type=bool,default=False)
parser.add_argument('--star_save_model_epoch',type=int,default=40)

parser.add_argument('--num_embeddings',type=int,default=50265)
parser.add_argument('--hidden_dim',type=int,default=200)
parser.add_argument('--num_channel',type=int,default=100)
parser.add_argument('--num_layers',type=int,default=6)
parser.add_argument('--dropout',type=float,default=0.3)
parser.add_argument('--filter_sizes',type=list,default=[2,3,5,7,11])
parser.add_argument('--top_p',type=int,default=3)
parser.add_argument('--att_head',type=int,default=2)
opts = parser.parse_args()

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

print(vars(opts))

CUDA_VISIBLE_DEVICES = [0, 1]

assert opts.epoch > opts.star_epoch

train_loader, test_loader = Dataset_process(opts)

GPANet = GPANet(opts)
# cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
device_ids = CUDA_VISIBLE_DEVICES
model = torch.nn.DataParallel(GPANet, device_ids=device_ids)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=opts.lr)
criterion = nn.CrossEntropyLoss().cuda()

if opts.resume != None:
    print('loading model checkpoint from %s..' % opts.resume)
checkpoint = torch.load(opts.resume)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['n_epoch'] + 1

loss_global = []

def train(epoch,opts):
    global loss_global
    for index, (inputs, target) in enumerate(train_loader):
        token = tokenizer(list(inputs), padding=True, return_tensors='pt')
        ids = token['input_ids']
        ids = ids.to(device)
        target = target.to(device)
        outputs = model(ids, opts)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_global.append(loss.item())
    if epoch > opts.star_epoch:
        save_model(model=model,optimizer=optimizer,n_epoch=epoch, opts=opts)


def test(opts):
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, label in test_loader:
            token = tokenizer(list(inputs), padding=True, return_tensors='pt')
            ids = token['input_ids']
            ids, label = ids.to(device), label.to(device)
            outputs = model(ids, opts)
            _, predicted = torch.max(outputs.data, dim=1)
            all_preds.extend(predicted)
            all_labels.extend(label)

        tensor_labels, tensor_preds = torch.tensor(all_labels), torch.tensor(all_preds)
        f1, precision, recall, tp, tn, fp, fn = all_metrics(tensor_labels, tensor_preds)
        print(
            'f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, loss: {:.4f}'.format(
                f1,
                precision,
                recall,
                sum(loss_global) / len(
                    loss_global)))
        print('tp: {:.4f}, tn: {:.4f}, fp: {:.4f}, fn: {:.4f}'.format(tp, tn, fp, fn))

        reslust = [f1, precision, recall, tp, tn, fp, fn, sum(loss_global) / len(loss_global)]
        save_data.append(reslust)


if __name__ == '__main__':
    save_data = []
    for epoch in range(opts.epoch):
        train(epoch,opts)
        print('-' * 50, epoch, '-' * 50)
    test(opts)
    save_df = pd.DataFrame(data=save_data,
                           columns=['f1', 'precision', 'recall', 'tp', 'tn', 'fp', 'fn',
                                    'loss'])
    save_df.to_excel(rf'./Result/{opts.project}.xlsx', index=False)