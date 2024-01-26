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

        diff_columns = self.__compare_columns(self.df_master, self.df_sample)
        diff_indices = self.__compare_index(self.df_master, self.df_sample)
        diff_strings = self.__compare_strings(self.df_master, self.df_sample)
        diff_numbers = self.__compare_numerical(self.df_master, self.df_sample, diff_columns, diff_indices, tol)

        if not any([len(diff_columns), len(diff_strings), len(diff_columns), len(diff_indices)]):
            print(f"Files {file1} and {file2} are identical within tol={tol}")
        else:
            print(f"Files {file1} and {file2} have differences")
            for df_name in ['diff_columns', 'diff_strings', 'diff_indices', 'diff_numbers']:
                if len(eval(df_name)):
                    print(f"{df_name}: {len(eval(df_name))}")
                    eval(df_name).to_csv(result + '/' + df_name + '.csv')
            print(f"See {result}/ for details")

        self.diffs = (diff_numbers, diff_strings, diff_columns, diff_indices)

    @staticmethod
    def __compare_strings(df_master, df_sample):
        # Make new dataframes with common columns and rows
        df1 = df_master[df_master.columns.intersection(df_sample.columns)]
        df1 = df1.drop(df_master.index.difference(df_sample.index))
        df2 = df_sample[df_sample.columns.intersection(df_master.columns)]
        df2 = df2.drop(df_sample.index.difference(df_master.index))

        df1_sorted = df1.select_dtypes(exclude=np.number).sort_index()
        df2_sorted = df2.select_dtypes(exclude=np.number).sort_index()
        df_compare = df1_sorted.compare(df2_sorted, align_axis=0, keep_shape=False, result_names=('master', 'sample'))

        df_compare.index.set_names('source', level=-1, inplace=True)
        return df_compare

    @staticmethod
    def __compare_columns(df_master, df_sample):
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

    @staticmethod
    def __compare_index(df_master, df_sample):
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

    @staticmethod
    def __compare_numerical(df_master, df_sample, df_columns, df_indices, tol):
        deleted_columns = df_columns[df_columns.status == 'deleted'].index
        deleted_indices = df_indices[df_indices.status == 'deleted'].index
        numerical_columns_master = df_master.select_dtypes(include=np.number).columns
        numerical_columns_sample = df_sample.select_dtypes(include=np.number).columns
        numerical_columns = numerical_columns_master.intersection(numerical_columns_sample)

        df_numerical_diff = (df_sample[numerical_columns] - df_master[numerical_columns])
        diff_mask = np.abs(df_numerical_diff.loc[df_master.index]) > tol

        df_diff = df_master.copy()
        df_diff[numerical_columns] = df_numerical_diff
        df_diff[~diff_mask] = 0

        if len(deleted_columns):
            df_diff[deleted_columns] = -df_diff[deleted_columns]
        if len(deleted_indices):
            df_diff.loc[deleted_indices, numerical_columns] = -df_master.loc[deleted_indices, numerical_columns]

        return df_diff
