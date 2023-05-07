from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class DataSet(Dataset):
    def __init__(self, data, label):
        super(DataSet, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], int(self.label[idx])

def Dataset_process(opts):

    with open(rf'/Data/{opts.project}/data.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()
    code_data = [x.split('<CODESPLIT>')[0] for x in data]
    code_label = [x.split('<CODESPLIT>')[1] for x in data]
    train_code, test_code, train_label, test_label = train_test_split(code_data, code_label, random_state=666,
                                                                      train_size=0.8)

    train_dataset = DataSet(train_code, train_label)
    test_dataset = DataSet(test_code, test_label)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader