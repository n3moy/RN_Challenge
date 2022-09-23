import os
import py7zr
# import pyarrow
import pandas as pd

ZIP_PATH = "../../data/archive/train_dataset.7z"
OUT_PATH = "../../data/processed"


def zip_to_parquet(zip_path: str, output_path: str = None):

    with py7zr.SevenZipFile(zip_path, "r") as z_file:
        filenames = z_file.getnames()
        selective_files = [f for f in filenames if "processed" in f and ".csv" in f]
        print(f"Amount of files to extract: {len(selective_files)}")
        tmp_path = os.path.join(output_path, "tmp")

        for ix, file_path in enumerate(selective_files):
            z_file.extract(targets=file_path, path=tmp_path)
            file_name = os.path.basename(file_path)
            new_name = file_name.replace(".csv", ".parquet")
            csv_file = pd.read_csv(os.path.join(tmp_path, file_path), low_memory=False)
            save_path = os.path.join(output_path, new_name)
            # print(f"Saved to {save_path}")
            csv_file.to_parquet(save_path)
            os.remove(os.path.join(tmp_path, file_path))

            z_file.reset()
            # Testing, may be removed
            if ix == 10:
                break


# Same performance, but without tmp folder and intermediate saving
def zip_to_parquet_v2(zip_path: str, output_path: str = None):

    with py7zr.SevenZipFile(zip_path, "r") as z_file:
        filenames = z_file.getnames()
        selective_files = [f for f in filenames if "processed" in f and ".csv" in f]
        print(f"Amount of files to extract: {len(selective_files)}")

        for ix, file_path in enumerate(selective_files):
            # print(z_file.read(file_path).items())
            res = z_file.read(file_path).items()
            file_name = os.path.basename(file_path)
            new_name = file_name.replace(".csv", ".parquet")

            for fname, bio in res:
                csv_file = pd.read_csv(bio, low_memory=False)
                save_path = os.path.join(output_path, new_name)
                # print(f"Saved to {save_path}")
                csv_file.to_parquet(save_path)

            z_file.reset()
            # Testing, may be removed
            if ix == 10:
                break


if __name__ == "__main__":
    zip_to_parquet(ZIP_PATH, OUT_PATH)
