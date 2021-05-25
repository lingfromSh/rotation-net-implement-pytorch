import bpy
import math
import os

import subprocess
from concurrent.futures import ProcessPoolExecutor

project_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir_path = os.path.join(project_dir_path, "data")
core_model_dir_path = os.path.join(data_dir_path, "ShapeNetCore.v2")

print("Project: {}".format(project_dir_path))
print("Data: {}".format(data_dir_path))
print("Core Model: {}".format(core_model_dir_path))

case_viewpoint_mapping = {
    "tiny": [
        (0, 0),
        (0, 90),
        (0, 180),
        (0, 270),
    ],
    "small": [
        (0, 0),
        (0, 30),
        (0, 60),
        (0, 90),
        (0, 120),
        (0, 150),
        (0, 180),
        (0, 210),
        (0, 240),
        (0, 270),
        (0, 300),
        (0, 330),
    ],
    "full": [
        (-60, 0),
        (-60, 30),
        (-60, 60),
        (-60, 90),
        (-60, 120),
        (-60, 150),
        (-60, 180),
        (-60, 210),
        (-60, 240),
        (-60, 270),
        (-60, 300),
        (-60, 330),
        (-30, 0),
        (-30, 30),
        (-30, 60),
        (-30, 90),
        (-30, 120),
        (-30, 150),
        (-30, 180),
        (-30, 210),
        (-30, 240),
        (-30, 270),
        (-30, 300),
        (-30, 330),
        (0, 0),
        (0, 30),
        (0, 60),
        (0, 90),
        (0, 120),
        (0, 150),
        (0, 180),
        (0, 210),
        (0, 240),
        (0, 270),
        (0, 300),
        (0, 330),
        (30, 0),
        (30, 30),
        (30, 60),
        (30, 90),
        (30, 120),
        (30, 150),
        (30, 180),
        (30, 210),
        (30, 240),
        (30, 270),
        (30, 300),
        (30, 330),
        (60, 0),
        (60, 30),
        (60, 60),
        (60, 90),
        (60, 120),
        (60, 150),
        (60, 180),
        (60, 210),
        (60, 240),
        (60, 270),
        (60, 300),
        (60, 330),
    ],
}


def move_camera(cam, vp, r):
    phi, theta = math.radians(vp[0]), math.radians(vp[1])
    x = r * math.cos(phi) * math.cos(theta)
    y = r * math.cos(phi) * math.sin(theta)
    z = r * math.sin(phi)
    cam.location = x, y, z


def generate_place_to_save(filepath):
    return os.path.join(
        data_dir_path,
        "output",
        os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(filepath)))),
        os.path.basename(os.path.dirname(os.path.dirname(filepath))),
        os.path.basename(os.path.dirname(filepath)),
        os.path.basename(filepath),
    )


def generate_multiview_images(dataset: str, model_dir_filepath: str):
    if bpy.data.objects.get("Cube"):
        bpy.data.objects.remove(bpy.data.objects["Cube"])

    for name in os.listdir(model_dir_filepath):
        path = os.path.join(model_dir_filepath, name)
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "models", "model_normalized.obj")):
                model_file_path = os.path.join(path, "models", "model_normalized.obj")
            elif os.path.exists(os.path.join(path, "model_normalized.obj")):
                model_file_path = os.path.join(path, "model_normalized.obj")
            else:
                continue

            bpy.ops.import_scene.obj(filepath=model_file_path)
            bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME", center="MEDIAN")
            model = bpy.context.selected_objects[0]
            model.location = 0, 0, 0
            max_dimension = max(model.dimensions)
            model.dimensions[0] = model.dimensions[0] / max_dimension
            model.dimensions[1] = model.dimensions[1] / max_dimension
            model.dimensions[2] = model.dimensions[2] / max_dimension
            bpy.data.scenes.update()

            camera = bpy.data.objects["Camera"]
            bpy.context.scene.camera = camera
            bpy.context.scene.render.engine = "CYCLES"
            bpy.context.scene.cycles.device = 'GPU'

            bpy.context.scene.render.resolution_x = 224
            bpy.context.scene.render.resolution_y = 224

            constraint = camera.constraints.new(type="TRACK_TO")
            constraint.target = model

            viewpoints = case_viewpoint_mapping[dataset]
            for idx, viewpoint in enumerate(viewpoints):
                move_camera(camera, viewpoint, 3)
                if os.path.exists(generate_place_to_save(model_file_path) + ".png" + f".{idx + 1}.png"):
                    continue
                bpy.context.scene.render.filepath = (
                        generate_place_to_save(model_file_path) + ".png" + f".{idx + 1}"
                )
                bpy.context.scene.render.film_transparent = 1
                bpy.ops.render.render(animation=False, write_still=True)
            bpy.data.objects.remove(model)


def main():
    with ProcessPoolExecutor() as executor:
        for name in os.listdir(core_model_dir_path):
            path = os.path.join(core_model_dir_path, name)
            if os.path.isdir(path):
                executor.submit(
                    subprocess.run,
                    "blender -b --python scripts/generate_multiview_images.py".split(),
                    env={
                        "CASE": "tiny",
                        "MODEL_DIR_FILEPATH": str(path),
                    },
                    cwd=project_dir_path,
                )


if __name__ == "__main__":
    import sys

    if sys.argv[0] == "blender":
        case = os.environ.get("CASE", "tiny")
        model_filepath = os.environ.get("MODEL_DIR_FILEPATH")

        generate_multiview_images(case, model_filepath)
    else:
        main()
