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

    def __init__(self, file1, file2, result, tol=1e-4):
        try:
            self.df_master = pd.read_csv(file1, index_col=0)
            self.df_sample = pd.read_csv(file2, index_col=0)
        except Exception as e:
            print(e)
            sys.exit(1)

        values = self.values(tol)
        strings = self.strings()
        columns = self.columns()
        index = self.index()

        if not any([len(values), len(strings), len(columns), len(index)]):
            id_str = f"Files {file1} and {file2} are identical within tol={tol}"
            print(id_str)
        else:
            print(f"Files {file1} and {file2} have differences")
            for name in ['values', 'strings', 'columns', 'index']:
                if len(eval(name)):
                    print(f"Number of {name} differences: {len(eval(name))}")
                    eval(name).to_csv(result + '/diff_' + name + '.csv')
                    if name == 'values':
                        self.diff_files(self.df_master, self.df_sample, tol).to_csv(result + '/diff_master.csv')
            print(f"See {result}/ for details")

        self.diffs = (values, strings, columns, index)

    def strings(self):
        return self.__compare(self.df_master, self.df_sample, dtype=object)

    def values(self, tol):
        return self.__compare(self.df_master, self.df_sample, dtype=np.number, tol=tol)

    def columns(self):
        return self.__compare_columns(self.df_master, self.df_sample)

    def index(self):
        return self.__compare_index(self.df_master, self.df_sample)

    def __compare(self, df_master, df_sample, dtype, tol=None):
        # Make new dataframes with common columns and rows
        df1 = df_master[df_master.columns.intersection(df_sample.columns)]
        df1 = df1.drop(df_master.index.difference(df_sample.index))
        df2 = df_sample[df_sample.columns.intersection(df_master.columns)]
        df2 = df2.drop(df_sample.index.difference(df_master.index))

        if dtype != np.number:
            df_compare = df1.select_dtypes(exclude=np.number).compare(df2.select_dtypes(exclude=np.number),
                                                                      align_axis=0,
                                                                      keep_shape=False,
                                                                      result_names=('master', 'sample'))
        else:
            numerical_columns = df1.select_dtypes(include=np.number).columns
            df_numerical_diff = np.abs(df2[numerical_columns] - df1[numerical_columns])
            diff_index = df_numerical_diff[(df_numerical_diff > tol).any(axis=1)].index

            df_numerical = df1.loc[diff_index][numerical_columns].compare(df2.loc[diff_index][numerical_columns],
                                                                          align_axis=0,
                                                                          keep_shape=False,
                                                                          result_names=('master', 'sample'))

            # select strings to use as labels
            df_temp = df1.loc[diff_index].select_dtypes(exclude=np.number).copy()
            df_labels = df_temp.compare(df_temp, align_axis=0, keep_shape=True, keep_equal=True,
                                        result_names=('master', 'sample'))

            df_compare = df_labels.merge(df_numerical, how='left', left_index=True, right_index=True)

        df_result = df_compare.rename_axis(index=['ID', 'source'])
        return df_result

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
        df_index_deleted = pd.Series(df_master.index.difference(df_sample.index), name='index')
        df_index_deleted = df_index_deleted.to_frame()
        df_index_deleted['status'] = 'deleted'
        df_index_deleted.set_index('index', inplace=True)

        df_index_added = pd.Series(df_sample.index.difference(df_master.index), name='index')
        df_index_added = df_index_added.to_frame()
        df_index_added['status'] = 'added'
        df_index_added.set_index('index', inplace=True)

        df_index = pd.concat([df_index_deleted, df_index_added])
        df_index.sort_index(inplace=True)
        df_index.rename_axis(index=['index'], inplace=True)

        return df_index

    @staticmethod
    def diff_files(df_master, df_sample, tol):
        numerical_columns = df_master[df_master.columns.intersection(df_sample.columns)].select_dtypes(
            include=np.number).columns

        # Set values to 99 where indices missing in sample
        df_numerical_diff = (df_sample.loc[df_master.index.isin(df_sample.index)][numerical_columns] - df_master[
            numerical_columns]).fillna(-99)
        diff_mask = np.abs(df_numerical_diff) > tol

        deleted_columns = df_master.columns.difference(df_sample.columns)
        df_master[deleted_columns] = -df_master[deleted_columns]
        df_master[numerical_columns] = df_numerical_diff
        df_master[~diff_mask] = 0

        return df_master
