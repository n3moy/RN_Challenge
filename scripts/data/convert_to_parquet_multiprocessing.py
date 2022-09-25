"""Работа с архивным файлом
   v.3 25-09-2022
"""

import os
import py7zr
import shutil
import asyncio

import threading
import multiprocessing

import pandas as pd
from pathlib import Path
from datetime import datetime

import logging


ZIP_PATH = "../../data/archive/test.7z"
OUT_PATH = "../../data/featured"

# ZIP_PATH = "test.7z"
# OUT_PATH = ".\out"
CLEAR_PATH = None   # путь к мусорным директориям

# ---
_work_list = []  # список извлечённых файлов
f_extract = False  # флаг работы процесса извлечения файлов
_debug = False

_mode = False  # True - использование multiprocessing, False - threading

logging.basicConfig(format='system:\t%(name)s   [%(asctime)s] [%(levelname)s]   %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


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


def convert(file_path) -> None:
   try:
      open_file = os.path.join(OUT_PATH, file_path)
      save_file = os.path.join(OUT_PATH, os.path.basename(file_path).replace(".csv", ".parquet"))
      csv_file = pd.read_csv(open_file, low_memory=False)
      csv_file.to_parquet(save_file)
      os.remove(open_file)
   except Exception:
      pass


# Same performance, but without tmp folder and intermediate saving
async def zip_to_parquet():
   """Процесс поочередного извлечения файлов из архива в точки открытия файла"""

   global _work_list, f_extract, CLEAR_PATH

   s1 = datetime.now()
   _count = 0
   
   with py7zr.SevenZipFile(ZIP_PATH, "r") as z_file:
      f_extract = True
      filenames = z_file.getnames()
      selective_files = [f for f in filenames if "processed" in f and ".csv" in f]
      if _debug:
         print(f"Amount of files to extract: {len(selective_files)}\n\n #  |  all   \n----|--------")
      async for file_path in async_list(selective_files, 1):
         if CLEAR_PATH is None:
            CLEAR_PATH = os.path.join(OUT_PATH, Path(file_path).parents._parts[0])
         s2 = datetime.now()
         try:
            z_file.extract(targets=file_path, path=OUT_PATH)
            z_file.reset()
         except Exception:
            pass
         else:
            _work_list.append(file_path)  # регистрация извлечённого файла
         if _debug:
            print(f"{_count+1:03} | {(datetime.now() - s2).total_seconds():6.2f}")

         # Testing, may be removed
         if _count == 20 and _debug:
               break
         _count += 1
         if _debug is False:
            print(f'count: {_count}', end="\r")
         await asyncio.sleep(0)
   if _debug:
      print(f"----^-------^--------\nTotal file {_count+1}, time: {(datetime.now() - s1).total_seconds():<8.3f} seconds\n")
   f_extract = False


async def file_processing():
   """Основной цикл обработки извлечённых файлов"""

   global _work_list, f_extract

   _tasks = []
   _f_break = False  # флаг прерывание цикла
   parent_conn, child_conn = multiprocessing.Pipe()
   while True:
      if _work_list:
         try:
            file_path = _work_list[0]  # первый в списке
            if _mode:
               p = multiprocessing.Process(target=convert, args=(file_path,))
               p.start()
               _tasks.append(p)
            else:
               t = threading.Thread(target=convert, args=(file_path,))
               t.start()
               _tasks.append(t)
         except Exception:
            pass
         del _work_list[0]  # удаляем изи списка
         await asyncio.sleep(0)
      elif len(_work_list) == 0 and f_extract is False:
         for x in _tasks:
            # проверка на полное завершение процесса
            x.join()
            _f_break = True
         
      if _f_break is True:
         break

if __name__ == "__main__":
   ioloop = asyncio.get_event_loop()
   tasks = [
   ioloop.create_task(zip_to_parquet()),  # извлечение файлов
   ioloop.create_task(file_processing())  # процесс конвертации файлов
   ]
   _st = datetime.now()
   ioloop.run_until_complete(asyncio.wait(tasks))
   ioloop.close()
   # ---
   if CLEAR_PATH is not None:
      # убираем все директории распаковки файлов
      try:
         shutil.rmtree(CLEAR_PATH)
         _tm = (datetime.now() - _st).total_seconds()
         print(f"\n>>> процесс конвертации завершён за {_tm:6.2f} seconds")
      except Exception as erDel:
         print(erDel)
