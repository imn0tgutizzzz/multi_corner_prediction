"""
File: data.py
Description: Contains functions to clean and treat the data,
             also to do some feature engineering.

Original Author: Anndress07    
Original Last update: 1/12/2024

Modified by:
    Josué María Jiménez Ramírez & Bryan Mora Porras

Modification Date:
    2026-06-28

Usage:
            Specify which function to call at the end of the code. 
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import os
print("Path Context: " + os.getcwd())


#%%
def data(data, save_csv_name):
    """
    Removes unnecesary columns in the dataset, creates Delta column (residual between actual and predicted delay)
    :param data: Dataset for the model
    :return: treated.csv, dataset with only the relevant columns for the model
    """
    df = pd.read_csv(data, index_col=0)

    df = df.reset_index(drop=True) # Creating index for later consistency
    df['row_id'] = df.index

    df.columns = df.columns.str.strip() # Delete " " empty spaces
    pd.set_option('display.max_columns', None)

        
    # If the df has Design column
    has_design = "Design" in df.columns

    if has_design: 
        col_names = ['row_id','Design','Previous_description','Description', 'Fanout', 'Cap', 'Slew', 'Delay', 'X_drive', 'Y_drive', 'X_sink',
                    'Y_sink', 'C_drive', 'C_sink', 'X_context', 'Y_context', 'σ(X)_context',
                    'σ(Y)_context', 'Drive_cell_size', 'Sink_cell_size', 'Label Delay']
    else:
        col_names = ['row_id','Previous_description','Description', 'Fanout', 'Cap', 'Slew', 'Delay', 'X_drive', 'Y_drive', 'X_sink',
                    'Y_sink', 'C_drive', 'C_sink', 'X_context', 'Y_context', 'σ(X)_context',
                    'σ(Y)_context', 'Drive_cell_size', 'Sink_cell_size', 'Label Delay']

    df = df[col_names]
    df = df.dropna()
    
    # Create Delta column: difference between actual and predicted delay
    df['Delta'] = df['Label Delay'] - df['Delay']
    
    # Keep Label Delay for now (used in corner analysis grouping)
    df = df.drop(columns=['Label Delay'])
    
    df.to_csv(save_csv_name, index=False)
    print(f'The file "{data}" was successfully modified and saved as:\n - {save_csv_name}')
    #print(df)
    #print(df.columns)



    # print(df.isna().sum())

def filtering(data):
    """
    Data visualization, not significant
    :param data: treated.csv, dataset with only the relevant columns for the model
    :return:
    """
    df = pd.read_csv(data)
    pd.set_option('display.max_columns', None)
    print(df.isna().sum())
    print(df.describe())
    df_2 = df[[' Fanout',' Cap',' Slew',' Delay','Label Delay','Delta']]
    df_3 = df[['X_drive','Y_drive','X_sink','Y_sink','C_drive','C_sink','X_context','Y_context','σ(X)_context','σ(Y)_context']]
    df_4 = df[['Drive_cell_size','Sink_cell_size']]

    gran_fanout = df[df['Delta']>0.65] # revisar 


    fig, ax = plt.subplots(figsize=(20, 20))
    """ 
    # First scatter plot: 'Fanout' vs. 'Delta'
    gran_fanout.plot(kind='scatter', x=' Fanout', y='Delta', color='blue', label='Fanout', ax=ax)
    # Second scatter plot: 'Another Fanout' vs. 'Delta'
    gran_fanout.plot(kind='scatter', x=' Cap', y='Delta', color='red', label='Cap', ax=ax)
    gran_fanout.plot(kind='scatter', x=' Slew', y='Delta', color='black', label='Slew', ax=ax)
    gran_fanout.plot(kind='scatter', x=' Delay', y='Delta', color='yellow', label='Delay', ax=ax)
    """
    # First scatter plot: 'Fanout' vs. 'Delta'
    gran_fanout.plot(kind='scatter', x='X_drive', y='Delta', color='blue', label='X_drive', ax=ax)
    # Second scatter plot: 'Another Fanout' vs. 'Delta'
    gran_fanout.plot(kind='scatter', x='Y_drive', y='Delta', color='red', label='Y_drive', ax=ax)
    gran_fanout.plot(kind='scatter', x='X_sink', y='Delta', color='black', label='X_sink', ax=ax)
    gran_fanout.plot(kind='scatter', x='Y_sink', y='Delta', color='yellow', label='Y_sink', ax=ax)

    ax.set_title('Scatter Plots for Multiple X-Variables against Delta')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Delta')
    ax.legend()
    plt.show()

def three_corners(data, corner):
    """
    Creates three different datasets depending on the corners of the transistors,
    slow, typical, fast based on the Delay column (transistor inherent timing)
    :param data: treated.csv, dataset with only the relevant columns for the model
    :param corner: decides the type of filtering to apply, can be "slow", "typical, or "fast"
    :return:    One of the following: slow.csv, typical.csv, fast.csv
    """
    df = pd.read_csv(data)
    df_filtered = pd.DataFrame(columns=df.columns)
    pd.set_option('display.max_columns', None)

    # Group by circuit parameters excluding 'Delay' (columns 0-3 and 5-17)
    group_cols = df.columns[:4].tolist() + df.columns[5:].tolist()

    if (corner == "fast"):
        df_filtered = df.loc[df.groupby(group_cols)['Delay'].idxmin()]
        df_filtered = df_filtered.reset_index(drop=True)
        df_filtered.to_csv("fast.csv", index=False)
    elif (corner == "slow"):
        df_filtered = df.loc[df.groupby(group_cols)['Delay'].idxmax()]
        df_filtered = df_filtered.reset_index(drop=True)
        print(df_filtered)
        df_filtered.to_csv("labels_slow.csv", index=False)
    # TODO: typical filtering not working properly.
    elif (corner == "typical"):
        grouped = df.groupby(group_cols)
        filtered_df = grouped.apply(get_quantile_row, quantile_value=0.5).reset_index(drop=True)
        df_filtered.to_csv("typical.csv", index=False)

# New function that splits between corner: slow, typical and fast
def split_by_corner(data, save_csv_name):
    """
    Splits the dataset into fast, typical, and slow corners
    by comparing rows that belong to the same path.

    A path is considered the same when:
        - Description is equal
        - Previous_Description is equal

    Then:
        - fast    -> minimum Delta
        - slow    -> maximum Delta
        - typical -> Delta closest to the mean Delta

    Generates:
        fast.csv
        slow.csv
        typical.csv
    """
    df = pd.read_csv(data)

    pd.set_option('display.max_columns', None)

    # Safety check
    required_cols = ['Description', 'Previous_description', 'Delay']

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    # Group rows belonging to the same path
    grouped = df.groupby(['Description', 'Previous_description'])

    fast_rows = []
    slow_rows = []
    typical_rows = []

    for _, group in grouped:

        # FAST -> smallest Delay
        fast_idx = group['Delay'].idxmin()
        fast_rows.append(df.loc[fast_idx])

        # SLOW -> largest Delay
        slow_idx = group['Delay'].idxmax()
        slow_rows.append(df.loc[slow_idx])

        # TYPICAL -> closest to average Delta
        mean_delay = group['Delay'].mean()

        typical_idx = (group['Delay'] - mean_delay).abs().idxmin()
        typical_rows.append(df.loc[typical_idx])

    # Convert to DataFrames
    fast_df = pd.DataFrame(fast_rows).reset_index(drop=True)
    slow_df = pd.DataFrame(slow_rows).reset_index(drop=True)
    typical_df = pd.DataFrame(typical_rows).reset_index(drop=True)

    # Save CSVs
    fast_df.to_csv(save_csv_name+'_fast.csv', index=False)
    slow_df.to_csv(save_csv_name+'_slow.csv', index=False)
    typical_df.to_csv(save_csv_name+'_typical.csv', index=False)

    print("Generated:")
    print(f' - {save_csv_name}_fast.csv')
    print(f' - {save_csv_name}_slow.csv')
    print(f' - {save_csv_name}_typical.csv')

def get_quantile_row(group, quantile_value=0.5):
    quantile_delay = group['Delay'].quantile(quantile_value)  # Get the specified quantile of ' Delay'
    # Get the row with the closest value to the quantile
    return group.iloc[(group['Delay'] - quantile_delay).abs().argsort()[:1]]


def test_three(data):
    df = pd.read_csv(data)
    pd.set_option('display.max_columns', None)

    # for row in range(len(df)):
    #     target_row = df.iloc[row,:].tolist()
    #     target_row = target_row[:-1]
    #
    #     # print(target_row)
    #     repeated = df[df.iloc[:, :16].eq(target_row).all(axis=1)]
    #     if (len(repeated) > 1):
    #         print(target_row)
    #         print(repeated)
    target_row2 = [0.0, 0.0, 0.23, 2.89, 953120.0, 696320.0, 953120.0, 696320.0, 0.004535, 0.124522,
                   946220.0, 690880.0, 0.0, 0.0, 2.0, 2.0]
    repeated = df[df.iloc[:, :16].eq(target_row2).all(axis=1)]
    print(repeated)

def remove_context_features(data_path):
    """
    Removes the features X_context and Y_context in both the training and testing data.
    Alters the file path name with the modified csv.
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """
    df_data = pd.read_csv(data_path)
    df_data = df_data.drop(columns=['X_context', 'Y_context'], errors='ignore')
    df_data.to_csv("modded_data.csv", index=False)
    NEW_DATA = "modded_data.csv"

    return NEW_DATA

def remove_std_dvt_context(data_path):
    """
    Removes the features σ(X)_context and σ(Y)_context in both the training and testing data.
    Alters the file path name with the modified csv.
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """

    df_data = pd.read_csv(data_path)
    df_data = df_data.drop(columns=['σ(X)_context', 'σ(Y)_context'], errors='ignore')
    df_data.to_csv("modded_data.csv", index=False)
    NEW_DATA = "modded_data.csv"

    return NEW_DATA


def calc_distance_parameter(data_path):
    """
    Calculates the Euclidean distance given X_drive, X_sink, Y_drive, Y_sink. Removes
    the aforementioned, adds a new parameter called "Distance"
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """

    df_data = pd.read_csv(data_path)
    df_data['Distance'] = np.sqrt((df_data['X_drive'] - df_data['X_sink']) ** 2 + (df_data['Y_drive'] - df_data['Y_sink']) ** 2)
    df_data = df_data.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])
    cols = df_data.columns.tolist()
    cols.remove('Distance')
    delay_idx = cols.index('Delay')
    new_cols_order = cols[:delay_idx + 1] + ['Distance'] + cols[delay_idx + 1:]
    df_data = df_data[new_cols_order]

    df_data.to_csv("modded_data.csv", index=False)
    NEW_DATA = "modded_data.csv"

    return NEW_DATA

def remove_context_features_two(train_data, test1, test2):
    """
    Removes the features X_context and Y_context in both the training and testing data.
    Alters the file path name with the modified csv.
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """
    df_train = pd.read_csv(train_data)
    df_train = df_train.drop(columns=['X_context', 'Y_context'], errors='ignore')
    df_train.to_csv("modded_train.csv", index=False)
    TRAINING_DATA = "modded_train.csv"
    df_test1 = pd.read_csv(test1)
    df_test1 = df_test1.drop(columns=['X_context', 'Y_context'], errors='ignore')
    df_test1.to_csv("modded_test1.csv", index=False)
    # print(f"desde el remove context, df_test: {df_test.columns}")
    TESTING1_DATA = "modded_test1.csv"

    df_test2 = pd.read_csv(test2)
    df_test2 = df_test2.drop(columns=['X_context', 'Y_context'], errors='ignore')
    df_test2.to_csv("modded_test2.csv", index=False)
    # print(f"desde el remove context, df_test: {df_test.columns}")
    TESTING2_DATA = "modded_test2.csv"

    return TRAINING_DATA, TESTING1_DATA, TESTING2_DATA

def remove_std_dvt_context_two(train_data, test1, test2):
    """
    Removes the features σ(X)_context and σ(Y)_context in both the training and testing data.
    Alters the file path name with the modified csv.
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """
    # if train_data != "modded_train.csv":
    #     TRAINING_DATA = C_TRAINING_DATA
    #     TESTING_DATA = C_TESTING_DATA
    df_train = pd.read_csv(train_data)
    df_train = df_train.drop(columns=['σ(X)_context', 'σ(Y)_context'], errors='ignore')
    df_train.to_csv("modded_train.csv", index=False)
    TRAINING_DATA = "modded_train.csv"

    df_test1 = pd.read_csv(test1)
    df_test1 = df_test1.drop(columns=['σ(X)_context', 'σ(Y)_context'], errors='ignore')
    df_test1.to_csv("modded_test1.csv", index=False)
    TESTING1_DATA = "modded_test1.csv"

    df_test2 = pd.read_csv(test2)
    df_test2 = df_test2.drop(columns=['σ(X)_context', 'σ(Y)_context'], errors='ignore')
    df_test2.to_csv("modded_test2.csv", index=False)
    TESTING2_DATA = "modded_test2.csv"
    return TRAINING_DATA, TESTING1_DATA, TESTING2_DATA

def calc_distance_parameter_two(train_data, test1, test2):
    """
    Calculates the Euclidean distance given X_drive, X_sink, Y_drive, Y_sink. Removes
    the aforementioned, adds a new parameter called "Distance"
    :param train_data: file name of the training data
    :param test_data: file name of the testing data
    :return: New path names including the modified csv
    """

    # if train_data != "modded_train.csv":
    #     TRAINING_DATA = C_TRAINING_DATA
    #     TESTING_DATA = C_TESTING_DATA
    df_train = pd.read_csv(train_data)
    df_train['Distance'] = np.sqrt(
        (df_train['X_drive'] - df_train['X_sink']) ** 2 + (df_train['Y_drive'] - df_train['Y_sink']) ** 2)
    df_train = df_train.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])
    cols = df_train.columns.tolist()
    cols.remove('Distance')
    delay_idx = cols.index(' Delay')
    new_cols_order = cols[:delay_idx + 1] + ['Distance'] + cols[delay_idx + 1:]
    df_train = df_train[new_cols_order]

    df_train.to_csv("modded_train.csv", index=False)
    TRAINING_DATA = "modded_train.csv"

    df_test1 = pd.read_csv(test1)
    df_test1['Distance'] = np.sqrt(
        (df_test1['X_drive'] - df_test1['X_sink']) ** 2 + (df_test1['Y_drive'] - df_test1['Y_sink']) ** 2)
    df_test1 = df_test1.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])
    cols = df_test1.columns.tolist()
    cols.remove('Distance')
    delay_idx = cols.index(' Delay')
    new_cols_order = cols[:delay_idx + 1] + ['Distance'] + cols[delay_idx + 1:]
    df_test1 = df_test1[new_cols_order]
    df_test1.to_csv("modded_test1.csv", index=False)
    TESTING1_DATA = "modded_test1.csv"

    df_test2 = pd.read_csv(test2)
    df_test2['Distance'] = np.sqrt(
        (df_test2['X_drive'] - df_test2['X_sink']) ** 2 + (df_test2['Y_drive'] - df_test2['Y_sink']) ** 2)
    df_test2 = df_test2.drop(columns=['X_drive', 'Y_drive', 'X_sink', 'Y_sink'])
    cols = df_test2.columns.tolist()
    cols.remove('Distance')
    delay_idx = cols.index(' Delay')
    new_cols_order = cols[:delay_idx + 1] + ['Distance'] + cols[delay_idx + 1:]
    df_test2 = df_test2[new_cols_order]
    df_test2.to_csv("modded_test2.csv", index=False)
    TESTING2_DATA = "modded_test2.csv"

    return TRAINING_DATA, TESTING1_DATA, TESTING2_DATA


"""

"""
#%%
if __name__ == "__main__":
    #data("CSVs/CSVs/2024_04_10/test_labels.csv") # Context Bryan
    #data("C:\\Users\\josue\\OneDrive\\Documentos\\Universidad de Costa Rica\\2026\\I-2026\\Proyecto_Electrico\\Delay-Delta-Hybrid-ML\\hybridModel-main\\source\\test_labels.csv") # Context Josue
    data("DeltaHybridModel-main\\source\\test_labels.csv", "treated_labels.csv") # Context Josue
    data("DeltaHybridModel-main\\source\\train.csv", "treated_labels_train.csv") # Context Josue
    split_by_corner("treated_labels.csv", "treated_labels") # Context Josue
    split_by_corner("treated_labels_train.csv", "treated_labels_train") # Context Josue

    # Unseen data
    data("DeltaHybridModel-main\\source\\test_designs.csv", "treated_test_designs.csv") # Context Josue
    split_by_corner("treated_test_designs.csv", "treated_test_designs")

    #data("../../../Data/2024_06_24/train.csv") # Context Josue
    #filtering('treated_labels.csv')
    # plotall('treated.csv')
    #three_corners('treated_labels.csv', 'slow')
    # test_three('typical.csv')

