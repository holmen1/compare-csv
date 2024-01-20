# compare-csv
A simple tool to compare two CSV files and output the differences.

## Usage
```python
from comparator import Comparator

compare = Comparator('file1.csv', 'file2.csv')   

compare.values(tol=1e-4))
```
ID, source, name, value1   
2,master,second,0.003   
  ,sample,second,0.004



You can also use the `Comparator` class in your own Python scripts. Here's an example:

```python
from comparator import Comparator

# Initialize the Comparator with two CSV files
compare = Comparator('file1.csv', 'file2.csv')

# Compare numerical values with a tolerance of 1e-6
df_num = compare.values(1e-6)

# Compare string values
df_str = compare.strings()

# Compare columns
df_columns = compare.columns()

# Compare indices
df_index = compare.index()
```

Each method returns a pandas DataFrame with the differences between the two CSV files.