import os
import random

import orjson
import argparse
from shutil import copyfile
from concurrent.futures import ProcessPoolExecutor

project_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir_path = os.path.join(project_dir_path, "data")
core_model_dir_path = os.path.join(data_dir_path, "ShapeNetCore.v2")
taxonomy_jsonpath = os.path.join(core_model_dir_path, "taxonomy.json")

name_category_mapping = {
    data.get("synsetId"): data
    for data in orjson.loads(open(taxonomy_jsonpath, "rb").read())
}

print("Project: {}".format(project_dir_path))
print("Data: {}".format(data_dir_path))
print("Core Model: {}".format(core_model_dir_path))
print("Taxonomy: {}".format(taxonomy_jsonpath))
# print("Name category Mapping: {}".format(name_category_mapping))

arg_parser = argparse.ArgumentParser("Split your dataset")
arg_parser.add_argument("-d", "--dst", default=os.path.join(data_dir_path, "shapeNet"), type=str, help="destination")


def move(src, dst):
    category = name_category_mapping[os.path.basename(src)]["name"]
    if not os.path.exists(os.path.join(dst, "train", category)):
        os.makedirs(os.path.join(dst, "train", category))
    if not os.path.exists(os.path.join(dst, "test", category)):
        os.makedirs(os.path.join(dst, "test", category))

    for name in os.listdir(src):
        is_train = random.random() > 0.2
        if is_train:
            d = os.path.join(dst, "train", category)
        else:
            d = os.path.join(dst, "test", category)

        if os.path.exists(os.path.join(src, name, "models")):
            path = os.path.join(src, name, "models")
        else:
            path = os.path.join(src, name)
        for png in os.listdir(path):
            p = os.path.splitext(png)[0].split(".")[-1]
            copyfile(src=os.path.join(path, png),
                     dst=os.path.join(d, f"{name}.{p}.png"))


def main():
    args = arg_parser.parse_args()
    with ProcessPoolExecutor() as executor:
        for name in os.listdir(os.path.join(data_dir_path, "output")):
            path = os.path.join(os.path.join(data_dir_path, "output"), name)
            if os.path.isdir(path):
                executor.submit(
                    move,
                    path,
                    args.dst
                )


if __name__ == "__main__":
    main()
