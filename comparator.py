# Compare CSV files
# Author: Mats Holm
import sys
import numpy as np
import pandas as pd


class Comparator:
    """
     A class to compare two CSV files.

     The Comparator class takes two CSV files as input, reads them into pandas DataFrames,
     and provides methods to compare the DataFrames in various ways. The differences are
     returned as pandas DataFrames.

     The class provides methods to compare the numerical and string values in the DataFrames,
     as well as the columns and indices. The comparison of numerical values takes a tolerance
     parameter to account for minor differences in floating point values.
     """

    def __init__(self, file1, file2, result, keys, tol=1e-4):
        try:
            self.df_master = pd.read_csv(file1, index_col=keys)
            self.df_sample = pd.read_csv(file2, index_col=keys)
        except Exception as e:
            print(e)
            sys.exit(1)

        self.tol = tol
        self.columns = self.columns()
        self.indices = self.indices()
        values = self.values(tol)
        strings = self.strings()

        if not any([len(values), len(strings), len(self.columns), len(self.indices)]):
            print(f"Files {file1} and {file2} are identical within tol={tol}")
        else:
            print(f"Files {file1} and {file2} have differences")
            for name in ['values', 'strings', 'self.columns', 'self.indices']:
                if len(eval(name)):
                    print(f"Number of {name} differences: {len(eval(name))}")
                    eval(name).to_csv(result + '/diff_' + name + '.csv')
                    if name == 'values':
                        self.diff_files().to_csv(result + '/diff_master.csv')
            print(f"See {result}/ for details")

        self.diffs = (values, strings, self.columns, self.indices)

    def strings(self):
        return self.__compare(self.df_master, self.df_sample, dtype=object)

    def values(self, tol):
        return self.__compare(self.df_master, self.df_sample, dtype=np.number, tol=tol)

    def columns(self):
        return self.__compare_columns(self.df_master, self.df_sample)

    def indices(self):
        return self.__compare_index(self.df_master, self.df_sample)

    def __compare(self, df_master, df_sample, dtype, tol=None):
        # Make new dataframes with common columns and rows
        df1 = df_master[df_master.columns.intersection(df_sample.columns)]
        df1 = df1.drop(df_master.index.difference(df_sample.index))
        df2 = df_sample[df_sample.columns.intersection(df_master.columns)]
        df2 = df2.drop(df_sample.index.difference(df_master.index))

        if dtype != np.number:
            df1_sorted = df1.select_dtypes(exclude=np.number).sort_index()
            df2_sorted = df2.select_dtypes(exclude=np.number).sort_index()
            df_compare = df1_sorted.compare(df2_sorted, align_axis=0, keep_shape=False,
                                            result_names=('master', 'sample'))
        else:
            numerical_columns = df1.select_dtypes(include=np.number).columns
            df_numerical_diff = np.abs(df2[numerical_columns] - df1[numerical_columns])
            diff_index = df_numerical_diff[(df_numerical_diff > tol).any(axis=1)].index.unique()

            df_numerical = df1.loc[diff_index][numerical_columns].compare(df2.loc[diff_index][numerical_columns],
                                                                          align_axis=0,
                                                                          keep_shape=False,
                                                                          result_names=('master', 'sample'))

            # select strings to use as labels
            df_temp = df1.loc[diff_index].select_dtypes(exclude=np.number).copy()
            df_labels = df_temp.compare(df_temp, align_axis=0, keep_shape=True, keep_equal=True,
                                        result_names=('master', 'sample'))

            df_compare = df_labels.merge(df_numerical, how='left', left_index=True, right_index=True)

        df_compare.index.set_names('source', level=-1, inplace=True)
        return df_compare

    def __compare_columns(self, df_master, df_sample):
        df_columns_deleted = df_master.columns.difference(df_sample.columns)
        df_columns_deleted = pd.DataFrame(df_columns_deleted, columns=['column'])
        df_columns_deleted['status'] = 'deleted'
        df_columns_deleted.set_index('column', inplace=True)

        df_columns_added = df_sample.columns.difference(df_master.columns)
        df_columns_added = pd.DataFrame(df_columns_added, columns=['column'])
        df_columns_added['status'] = 'added'
        df_columns_added.set_index('column', inplace=True)

        df_columns = pd.concat([df_columns_deleted, df_columns_added])
        df_columns.sort_index(inplace=True)
        df_columns.rename_axis(index=['column'], inplace=True)

        return df_columns

    def __compare_index(self, df_master, df_sample):
        df_index_deleted = df_master.index.difference(df_sample.index)
        df_index_deleted = pd.DataFrame(df_index_deleted, columns=['index'])
        df_index_deleted['status'] = 'deleted'
        df_index_deleted.set_index('index', inplace=True)

        df_index_added = df_sample.index.difference(df_master.index)
        df_index_added = pd.DataFrame(df_index_added, columns=['index'])
        df_index_added['status'] = 'added'
        df_index_added.set_index('index', inplace=True)

        df_index = pd.concat([df_index_deleted, df_index_added])
        df_index.sort_index(inplace=True)
        df_index.rename_axis(index=['index'], inplace=True)

        return df_index

    def diff_files(self):
        nan_num = -99.99

        numerical_columns_master = self.df_master.select_dtypes(include=np.number).columns
        numerical_columns_sample = self.df_sample.select_dtypes(include=np.number).columns
        numerical_columns = numerical_columns_master.intersection(numerical_columns_sample)

        df_numerical_diff = (self.df_sample[numerical_columns] - self.df_master[numerical_columns]).fillna(nan_num)
        diff_mask = np.abs(df_numerical_diff) > self.tol

        added_columns = numerical_columns_sample.intersection(self.columns[self.columns.status == 'added'].index)
        deleted_columns = numerical_columns_master.intersection(self.columns[self.columns.status == 'deleted'].index)
        deleted_indices = self.df_master.index.intersection(self.indices[self.indices == 'deleted'].index)

        df_diff = self.df_master.copy()
        df_diff[numerical_columns] = df_numerical_diff
        df_diff[~diff_mask] = 0

        if len(added_columns):
            df_diff[added_columns] = self.df_sample[added_columns]
            if len(deleted_indices): df_diff.loc[deleted_indices, added_columns] = nan_num
        if len(deleted_columns):
            df_diff[deleted_columns] = -df_diff[deleted_columns]
            if len(deleted_indices): df_diff.loc[deleted_indices, deleted_columns] = nan_num

        return df_diff
