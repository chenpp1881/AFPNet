import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from transformers import AutoTokenizer
from GPANet import GPANet
import argparse
from Dataset_process import Dataset_process


parser = argparse.ArgumentParser()
parser.add_argument('--is_train', type=bool, default=True)

# parameters
parser.add_argument('--project', type=str, default='reentrancy',
                    choices=['reentrancy', 'timestamp', 'loops', 'reentrancy_cleaned', 'timestamp_cleaned'])
parser.add_argument('--resume', type=str)
parser.add_argument('--seq_model', type=str, default='Att', choices=['Att', 'RNN', 'LSTM', 'RNN'])
parser.add_argument('--return_indecs', type=bool, default=False)
parser.add_argument('--fail_code_path', type=str)
parser.add_argument('--success_code_path', type=str)
parser.add_argument('--success_n_code_path', type=str)

parser.add_argument('--num_embeddings', type=int, default=50265)
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--num_channel', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--filter_sizes', type=list, default=[2,3,5,7,11])
parser.add_argument('--top_p', type=int, default=3)
parser.add_argument('--att_head', type=int, default=2)

opts = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

CUDA_VISIBLE_DEVICES = [0]

_, test_loader = Dataset_process(opts)

GPANet = GPANet(opts)

# cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
device_ids = CUDA_VISIBLE_DEVICES
model = torch.nn.DataParallel(GPANet, device_ids=device_ids)
model.to(device)

if opts.resume != None:
    print('loading model checkpoint from %s..' % opts.resume)
    checkpoint = torch.load(opts.resume)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['n_epoch'] + 1


def count_token(indecs, batch_index, ids, filter_sizes):
    # i 4,200,3
    count_str = ''
    for i, f in zip(indecs, filter_sizes):
        for x in i[batch_index, :, :].tolist():
            for y in x:
                count_str += ' '
                count_str += tokenizer.decode(ids[batch_index, y:y + f].tolist())
    return count_str



def test(opts):
    all_labels = []
    all_preds = []
    fail_code_path = opts.fail_code_path
    success_code_path = opts.success_code_path
    success_n_code_path = opts.success_n_code_path
    f = open(fail_code_path, 'w', encoding='utf-8')
    s = open(success_code_path, 'w', encoding='utf-8')
    n = open(success_n_code_path, 'w', encoding='utf-8')
    with torch.no_grad():
        for index, (inputs, label) in enumerate(test_loader):
            token = tokenizer(list(inputs), padding=True, return_tensors='pt')
            ids = token['input_ids']
            ids, label = ids.to(device), label.to(device)
            outputs, indices = model(ids)
            _, predicted = torch.max(outputs.data, dim=1)
            all_preds.extend(predicted)
            all_labels.extend(label)
            fail = label == predicted

            for x, y in enumerate(fail):
                if y == False:
                    f.write('<CODESPLIT>'.join(
                        [str(label[x].tolist()), inputs[x], count_token(indices, x, ids, filter_sizes=[int(i) for i in
                                                                                                       opts.filter_sizes.split(
                                                                                                           ',')]).replace(
                            '\n', '')]))
                    f.write('\n')
                elif y == True and label[x].tolist() == 1:
                    s.write('<CODESPLIT>'.join(
                        [str(label[x].tolist()), inputs[x],
                         count_token(indices, x, ids, [int(i) for i in opts.filter_sizes.split(',')]).replace('\n',
                                                                                                              '')]))
                    f.write('\n')
                elif y == True and label[x].tolist() == 0:
                    n.write('<CODESPLIT>'.join(
                        [str(label[x].tolist()), inputs[x],
                         count_token(indices, x, ids, [int(i) for i in opts.filter_sizes.split(',')]).replace('\n',
                                                                                                              '')]))
                    f.write('\n')

if __name__ == '__main__':
    test(opts)
