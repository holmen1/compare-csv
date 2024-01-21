# compare-csv
A simple tool to compare two CSV files and output the differences.

## Usage
```
from comparator import Comparator

compare = Comparator(file1, file2, result_folder, tol=1e-4)
```

```
Files data/master.csv and data/sample.csv have differences
Number of value differences: 2
Number of string differences: 2
Number of column differences: 2
Number of index differences: 2
See data/ for details
```

```
values_df, strings_df, columns_df, index_df = compare.diffs
```

```
values_df
```

|ID| source|      name | value1 |
|---:|---:|----------:|-------:|
|2|master|    second |  0.003 |
|2|sample|second|  0.004 |
