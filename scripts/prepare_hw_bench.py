import argparse

import pickle
import pandas as pd

from naslib.search_spaces.nasbench201.conversions import convert_str_to_op_indices



def main(in_path="../data/", out_path="../data/"):
    with open(f"{in_path}/HW-NAS-Bench-v1_0.pickle", "rb") as f:
        hw = pickle.load(f)

    hw = hw["nasbench201"]

    tasks =  hw.keys()

    for task in tasks:
        hw_task = hw[task]
        nets = [net["arch_str"] for net in hw_task["config"]]
        nets = [convert_str_to_op_indices(net) for net in nets]

        hw_task["net"] = nets
        del hw_task["config"]

        df = pd.DataFrame(hw_task).set_index("net")
        out_filename =f"{out_path}/hw_{task}.csv" 
        print(f"Saving {out_filename}")
        df.to_csv(out_filename)
        print("Saved.")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts data from HW Bench pickle file to CSV files."
    )

    parser.add_argument('--in_path', default="../data/", type=str,
                        help="Path to the directory where to find HW-NAS-Bench-v1.0.pickle.")
    parser.add_argument('--out_path', default="../data/", type=str,
                        help="Path to the directory where to save CSV files.")

    args = vars(parser.parse_args())
    main(**args)

