import process_data
import pandas as pd


def dist_dict(train):
    weights = [v for k, v in train.items()]
    sections = [0, 50, 200, 500, 1500, 4000, 8000]
    group_names = ['[0,50)', '[50,200)', '[200,500)', '[500,1500)', '[1500, 4000)',
                   '[4000, 8000)']
    cuts = pd.cut(weights, sections, labels=group_names)
    counts = pd.value_counts(cuts)
    print(dict(counts))
    for k, v in dict(counts).items():
        print(str(k) + "\t" + str(v))
    cuts.value_counts().plot(kind='bar')


tag_dicts = process_data.get_tags_count()
train = tag_dicts['train.txt']
train.pop('O')
test = tag_dicts['test.txt']
test.pop('O')
dev = tag_dicts['dev.txt']
dev.pop('O')
# train.pop('B_Time')
# train.pop('I_Time')
# train.pop('I_FamilyConflict')
dist_dict(train)
# dist_dict(test)
# dist_dict(dev)
