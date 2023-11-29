import glob
import pandas as pd 
import matplotlib.pyplot as plt
import argparse
from huggingface_hub import upload_file

import sys 
sys.path.append(".")
from utils import collate_csv # noqa: E402

REPO_ID = "sayakpaul/sample-datasets"

def prepare_plot(df, args):
    cmap = plt.cm.get_cmap('viridis', len(df)) 

    plt.figure(figsize=(10, 6))

    for i in range(len(df)):
        color = cmap(i)
        bar = plt.bar(i, df.iloc[i]['time (secs)'], color=color, label=f'{df.iloc[i]["run_compile"]}, {df.iloc[i]["compile_mode"]}')
        plt.text(
            bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height(),
            f'{df.iloc[i]["time (secs)"]:.2f}', 
            ha='center', va='bottom'
        )
        
    plt.ylabel('Time (secs)')
    plt.title('Benchmarking Results')
    plt.xticks(rotation=45)
    plt.legend(title='Run Compile / Compile Mode', bbox_to_anchor=(1.05, 1), loc='upper left')  
    
    plt.tight_layout()
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.savefig(args.plot_file_path)

    if args.push_to_hub:
        upload_file(repo_id=REPO_ID, path_in_repo=args.plot_file_path, path_or_fileobj=args.plot_file_path, repo_type="dataset")
        print(f"Plot successfully uploaded. Find it here: https://huggingface.co/datasets/{REPO_ID}/blob/main/{args.plot_file_path}")

def main(args):
    all_csvs = sorted(glob.glob(f"{args.base_path}/*.csv"))
    collate_csv(all_csvs, args.final_csv_filename)

    df = pd.read_csv(args.final_csv_filename)
    prepare_plot(df, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=".")
    parser.add_argument("--final_csv_filename", type=str, default="collated_results.csv")
    parser.add_argument("--plot_file_path", type=str, default="results.png")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    main(args)
    


