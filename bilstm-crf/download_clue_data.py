""" Script for downloading all CLUE data.
For licence information, see the original dataset information links
available from: https://www.cluebenchmarks.com/
Example usage:
  python download_clue_data.py --data_dir data --tasks all
"""

import os
import sys
import argparse
import urllib.request
import zipfile

TASKS = ['cluener']

TASK2PATH = {
    'cluener':'https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip'
}

def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    data_file = os.path.join(data_dir, "%s_public.zip" % task)
    save_dir = os.path.join(data_dir,task)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(save_dir)
    os.remove(data_file)
    print(f"\tCompleted! Downloaded {task} data to directory {save_dir}")

def get_tasks(task_names):
    task_names = task_names.split(",")
    if "all" in task_names:
        tasks = TASKS
    else:
        tasks = []
        for task_name in task_names:
            assert task_name in TASKS, "Task %s not found!" % task_name
            tasks.append(task_name)
    return tasks

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_dir", help="directory to save data to", type=str, default="./dataset"
    )
    parser.add_argument(
        "-t",
        "--tasks",
        help="tasks to download data for as a comma separated string",
        type=str,
        default="cluener",
    )
    args = parser.parse_args(arguments)

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_tasks(args.tasks)

    for task in tasks:
        download_and_extract(task, args.data_dir)#!

if __name__ == "__main__":
    #python download_clue_data.py --data_dir=./dataset --tasks=cluener
    sys.exit(main(sys.argv[1:]))