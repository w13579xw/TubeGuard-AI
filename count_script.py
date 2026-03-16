import csv
import collections
import os

aug_csv = 'data/defect_test/augmented.csv'
counts = collections.Counter()
with open(aug_csv, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if len(row) < 2: continue
        fname = row[0]
        if fname.startswith('aug_'):
            parts = fname.split('_')
            if len(parts) >= 2:
                atype = parts[1]
                counts[atype] += 1
        else:
            counts['original'] += 1

print('File type counts in augmented.csv:', counts)
