import os
import glob
from pathlib import Path

from bagpy import bagreader
import pandas as pd
import numpy as np
from tqdm import tqdm


def process_bag_files(bagfiles_path):
    # check if the given path exists
    if not os.path.exists(bagfiles_path):
        return "Given path does not exist"

    # get all bag files in the directory
    bagfiles = glob.glob(bagfiles_path + "/*.bag")
    bagfiles.sort()

    # read each bag file and get the topics
    # this loop creates record.csv files in each bag file directory
    for file in tqdm(bagfiles, desc="bag files"):
        try:
            b = bagreader(file, verbose=False)
            for t in b.topics:
                data = b.message_by_topic(t)
        except:
            print(f"Error reading {file}")
            continue

    return f"total bag files: {len(bagfiles)}"


def process_rec_csv(df_st1, poscols, velcols, effcols):
    # drop header columns and joint_names column
    df_st1.drop(df_st1.columns[1:6], axis=1, inplace=True)

    # appply eval() to convert string to list
    df_st1["joint_states.position"] = df_st1["joint_states.position"].apply(
        lambda x: eval(x)
    )
    df_st1["joint_states.velocity"] = df_st1["joint_states.velocity"].apply(
        lambda x: eval(x)
    )
    df_st1["joint_states.effort"] = df_st1["joint_states.effort"].apply(
        lambda x: eval(x)
    )

    # convert list of strings to list of floats
    df_st1[
        ["joint_states.position", "joint_states.velocity", "joint_states.effort"]
    ] = df_st1[
        ["joint_states.position", "joint_states.velocity", "joint_states.effort"]
    ].apply(
        lambda x: [np.float64(i) for i in x]
    )

    # create new columns for each joint_states
    df_st1[poscols] = pd.DataFrame(
        df_st1["joint_states.position"].tolist(), index=df_st1.index
    )
    df_st1[velcols] = pd.DataFrame(
        df_st1["joint_states.velocity"].tolist(), index=df_st1.index
    )
    df_st1[effcols] = pd.DataFrame(
        df_st1["joint_states.effort"].tolist(), index=df_st1.index
    )

    # drop the old joint_states columns
    df_st1.drop(
        ["joint_states.position", "joint_states.velocity", "joint_states.effort"],
        axis=1,
        inplace=True,
    )

    # shift failure_type.data and fault_flag.data to the last columns
    df_st1 = df_st1[
        [c for c in df_st1 if c not in ["failure_type.data", "fault_flag.data"]]
        + ["fault_flag.data", "failure_type.data"]
    ]

    return df_st1


def create_csv_files(bagfiles_path, csv_dir_path):
    rfiles = glob.glob(bagfiles_path + "*/record.csv")
    rfiles.sort()

    if len(rfiles) == 0:
        return "No record.csv files found in the given directory"

    poscols = [f"joint_position_{i}" for i in range(9)]
    velcols = [f"joint_velocity_{i}" for i in range(9)]
    effcols = [f"joint_effort_{i}" for i in range(9)]

    # create a new directory
    if not os.path.exists(csv_dir_path):
        os.makedirs(csv_dir_path)

    # rename each record.csv file by its parent directory name
    for file in tqdm(rfiles, desc="csv files"):
        df_rec = pd.read_csv(file)
        fail_count = df_rec["failure_type.data"].value_counts()
        if fail_count.shape[0] > 1:
            df_rec = process_rec_csv(df_rec, poscols, velcols, effcols)
            parent_name = file.split("/")[-2]
            df_rec.to_csv(f"{csv_dir_path}/{parent_name}.csv", index=False)

    # get total number of csv files created
    csvfiles = glob.glob(csv_dir_path + "/*.csv")

    return f"{len(csvfiles)} CSV files created successfully"
