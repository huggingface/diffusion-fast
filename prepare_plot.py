import glob
import pandas as pd 
import matplotlib.pyplot as plt
import argparse

import sys 
sys.path.append(".")
from utils import collate_csv

def prepare_plot(df, args):
    cmap = plt.cm.get_cmap('viridis', len(df))  # Using 'viridis' colormap, change as needed

    plt.figure(figsize=(10, 6))

    for i in range(len(df)):
        color = cmap(i)
        plt.bar(df.iloc[i]['pipeline_cls'], df.iloc[i]['time (secs)'], color=color, label=f'{df.iloc[i]["run_compile"]}, {df.iloc[i]["compile_mode"]}')

    plt.xlabel('Pipeline Class')
    plt.ylabel('Time (secs)')
    plt.title('Benchmarking Results')
    plt.xticks(rotation=45)
    plt.legend(title='Run Compile / Compile Mode', bbox_to_anchor=(1.05, 1), loc='upper left')  # Moving legend outside the plot
    plt.tight_layout()
    plt.savefig(args.plot_file_path)


def main(args):
    all_csvs = sorted(glob.glob(f"{args.base_path}/*.csv"))
    collate_csv(all_csvs, args.final_csv_filename)

    df = pd.read_csv(args.final_csv_filename)
    prepare_plot(df, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=".")
    parser.add_argument("--final_csv_filename", type=str, default="collated_results.csv")
    parser.add_argument("--plot_file_path", action="results.png")
    args = parser.parse_args()

    main(args)
    


