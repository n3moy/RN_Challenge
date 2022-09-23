import asyncio
# import subprocess
# from multiprocessing import Process
import os
import py7zr
# from pyunpack import Archive
# import pyarrow
import pandas as pd
from datetime import datetime

ZIP_PATH = "../../data/archive/train_dataset.7z"
OUT_PATH = "../../data/processed"


class async_list():
    """Формирование асинхронного итератора списка

        obj     - итерируемый объект

        type_it - тип объекта:
            = 0 предоставлен итератор
            = 1 предоставлен список
                (нужно перевести в итератор)
    """

    def __init__(self, obj, type_it):
        if type_it == 0:
            self._it = obj
        elif type_it == 1:
            self._it = iter(obj)
        else:
            self._it = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            value = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        return value


async def read_file(file_path, z_file, output_path, ix):
    s2 = datetime.now()
    # res = z_file.read(file_path).items()
    res = await read_zip(file_path, z_file)
    file_name = os.path.basename(file_path)
    new_name = file_name.replace(".csv", ".parquet")

    for fname, bio in res:
        s3 = datetime.now()
        csv_file = pd.read_csv(bio, low_memory=False)
        save_path = os.path.join(output_path, new_name)
        # print(f"Saved to {save_path}")
        csv_file.to_parquet(save_path)
    z_file.reset()
    print(f"{ix + 1} block time: {(datetime.now() - s2).total_seconds()} / "
          f"{(datetime.now() - s3).total_seconds()}")
    await asyncio.sleep(0)


async def read_zip(file_path, z_file):
    return z_file.read(file_path).items()


# Same performance, but without tmp folder and intermediate saving
async def zip_to_parquet(zip_path: str, output_path: str = None):

    s1 = datetime.now()
    with py7zr.SevenZipFile(zip_path, "r") as z_file:
        filenames = z_file.getnames()
        selective_files = [f for f in filenames if "processed" in f and ".csv" in f]
        print(f"Amount of files to extract: {len(selective_files)}")
        ix = 0
        tasks = [asyncio.ensure_future(read_file(file_path, z_file, output_path, ix)) for ix, file_path in enumerate(selective_files) if ix < 2]
        await asyncio.wait(tasks)

        # async for file_path in async_list(selective_files, 1):
        #     # print(z_file.read(file_path).items())
        #     # C:\py\esp-failures\scripts\data\convert_to_parquet.py
        #     p = Process(target=read_file, args=(file_path, z_file, output_path, ix), name=f'Process {ix}')
        #     p.start()
            # tasks = subprocess.Popen([read_file, file_path, z_file, output_path, ix])
            # tasks.wait()
            # await read_file(file_path, z_file, output_path, ix)
            # await asyncio.sleep(0)
            # ---
            # s2 = datetime.now()
            # # res = z_file.read(file_path).items()
            # res = await read_zip(file_path, z_file)
            # file_name = os.path.basename(file_path)
            # new_name = file_name.replace(".csv", ".parquet")
            #
            # for fname, bio in res:
            #     s3 = datetime.now()
            #     csv_file = pd.read_csv(bio, low_memory=False)
            #     save_path = os.path.join(output_path, new_name)
            #     # print(f"Saved to {save_path}")
            #     csv_file.to_parquet(save_path)
            # z_file.reset()
            # print(f"{ix+1} block time: {(datetime.now() - s2).total_seconds()} / "
            #       f"{(datetime.now() - s3).total_seconds()}")
            # ---

            # Testing, may be removed
            # if ix == 1:
            #     break
            # ix += 1

    print(f"Total file {ix+1} time: {(datetime.now() - s1).total_seconds()}")


def zip_to_parquet_v2(zip_path: str, output_path: str = None):

    s1 = datetime.now()
    with py7zr.SevenZipFile(zip_path, "r") as z_file:
        filenames = z_file.getnames()
        selective_files = [f for f in filenames if "processed" in f and ".csv" in f]
        print(f"Amount of files to extract: {len(selective_files)}")
        print(f"Time to open 7zip: {(datetime.now() - s1).total_seconds()}")
        for ix, file_path in enumerate(selective_files):
            # print(z_file.read(file_path).items())
            s2 = datetime.now()
            res = z_file.read(file_path).items()
            file_name = os.path.basename(file_path)
            new_name = file_name.replace(".csv", ".parquet")
            print(f"Time to read {ix+1} file: {(datetime.now() - s2).total_seconds()}")
            for fname, bio in res:
                csv_file = pd.read_csv(bio, low_memory=False)
                save_path = os.path.join(output_path, new_name)
                # print(f"Saved to {save_path}")
                csv_file.to_parquet(save_path)

            z_file.reset()
            # Testing, may be removed
            if ix == 1:
                break

    print(f"Total files {ix + 1} time: {(datetime.now() - s1).total_seconds()}")


def zip_to_parquet_v3(zip_path: str, output_path: str = None):

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


if __name__ == "__main__":
    # asyncio.run(zip_to_parquet(ZIP_PATH, OUT_PATH))
    zip_to_parquet_v2(ZIP_PATH, OUT_PATH)
