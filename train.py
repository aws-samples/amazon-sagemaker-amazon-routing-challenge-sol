import argparse
import time

import joblib
import pandas as pd

from aro.model.ppm import build_ppm_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="check zone sequence")
    parser.add_argument(
        "--train_zdf_fn",
        default='data/Final_March_15_Data/zone_list/actual_zone-train.csv',
        help="location of the actual sequence with zone information"
    )
    parser.add_argument(
        "--ppm_model_fn", 
        default="aro_ppm_train_model.joblib", 
        type=str, 
        help="File name of the PPM model"
    )
    args = parser.parse_args()
    train_zdf = pd.read_csv(args.train_zdf_fn)

    stt = time.time()
    ppm_model = build_ppm_model(train_zdf, 5, gt_strictly_set=True)
    dur = time.time() - stt
    print(f'Time to train PPM = {dur:.3f} seconds')
    print('Saving the model to the current directory ...', end='')
    joblib.dump(ppm_model, args.ppm_model_fn)
    print('done')