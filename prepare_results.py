import argparse
import glob
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from huggingface_hub import upload_file


sys.path.append(".")
from utils.benchmarking_utils import collate_csv  # noqa: E402


REPO_ID = "sayakpaul/sample-datasets"


def prepare_plot(df, args):
    # Drop the columns that are not needed
    columns_to_drop = [
        "batch_size",
        "num_inference_steps",
        "pipeline_cls",
        "ckpt_id",
        "upcast_vae",
        "memory (gbs)",
        "actual_gpu_memory (gbs)",
        "tag",
    ]
    df_filtered = df.drop(columns=columns_to_drop)
    df_filtered[["quant"]] = df_filtered[["do_quant"]].fillna("None")
    df_filtered.drop(columns=["do_quant"], inplace=True)

    # Create a new column to consolidate settings into a readable format
    df_filtered["settings"] = df_filtered.apply(
        lambda row: ", ".join([f"{col}-{row[col]}" for col in df_filtered.columns if col != "time (secs)"]), axis=1
    )
    df_filtered["formatted_settings"] = df_filtered["settings"].str.replace(", ", "\n", regex=False)
    df_filtered.loc[0, "formatted_settings"] = "default"

    # Generating the plot with matplotlib directly for better control
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")

    # Calculate the number of unique settings for bar positions
    n_settings = len(df_filtered["formatted_settings"].unique())
    bar_positions = range(n_settings)

    # Choose a color palette
    palette = sns.color_palette("husl", n_settings)

    # Plot each bar manually
    bar_width = 0.25  # Width of the bars
    for i, setting in enumerate(df_filtered["formatted_settings"].unique()):
        # Filter the dataframe for each setting and get the mean time
        mean_time = df_filtered[df_filtered["formatted_settings"] == setting]["time (secs)"].mean()
        plt.bar(i, mean_time, width=bar_width, align="center", color=palette[i])

        # Add the text above the bars
        plt.text(i, mean_time + 0.01, f"{mean_time:.2f}", ha="center", va="bottom", fontsize=14, fontweight="bold")

    # Set the x-ticks to correspond to the settings
    plt.xticks(bar_positions, df_filtered["formatted_settings"].unique(), rotation=45, ha="right", fontsize=10)

    plt.ylabel("Time in Seconds", fontsize=14, labelpad=15)
    plt.xlabel("Settings", fontsize=14, labelpad=15)
    plt.title(args.plot_title, fontsize=18, fontweight="bold", pad=20)

    # Adding horizontal gridlines for better readability
    plt.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust the top and bottom

    plot_path = args.plot_title.replace(" ", "_") + ".png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)

    if args.push_to_hub:
        upload_file(repo_id=REPO_ID, path_in_repo=plot_path, path_or_fileobj=plot_path, repo_type="dataset")
        print(
            f"Plot successfully uploaded. Find it here: https://huggingface.co/datasets/{REPO_ID}/blob/main/{args.plot_file_path}"
        )

    # Show the plot
    plt.show()


def main(args):
    all_csvs = sorted(glob.glob(f"{args.base_path}/*.csv"))
    collate_csv(all_csvs, args.final_csv_filename)

    if args.push_to_hub:
        upload_file(
            repo_id=REPO_ID,
            path_in_repo=args.final_csv_filename,
            path_or_fileobj=args.final_csv_filename,
            repo_type="dataset",
        )
        print(
            f"CSV successfully uploaded. Find it here: https://huggingface.co/datasets/{REPO_ID}/blob/main/{args.final_csv_filename}"
        )

    if args.plot_title is not None:
        df = pd.read_csv(args.final_csv_filename)
        prepare_plot(df, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default=".")
    parser.add_argument("--final_csv_filename", type=str, default="collated_results.csv")
    parser.add_argument("--plot_title", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    main(args)
