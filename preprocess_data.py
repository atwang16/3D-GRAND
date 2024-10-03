import argparse
import json
import os

import trimesh
import numpy as np
import trimesh.transformations as tf


# adjust bright factor to increase or decrease the brightness of the colors
def convert_ply_to_format(ply_file: str, output_file: str, bright_factor: float = 1.5):
    # Load the PLY file
    mesh = trimesh.load(ply_file)

    # Define the rotation matrix to swap y and z axes
    angle = -np.pi / 2  # 90 degrees
    axis = [1, 0, 0]  # Rotate around x-axis
    R = tf.rotation_matrix(angle, axis)

    # Apply the rotation to the mesh
    mesh.apply_transform(R)

    # Adjust the brightness of vertex colors if they exist
    if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
        vertex_colors = np.asarray(mesh.visual.vertex_colors)[:, :3]  # Ignore alpha channel if present
        brightened_colors = np.clip(vertex_colors * bright_factor, 0, 255).astype(np.uint8)
        mesh.visual.vertex_colors = brightened_colors

    # Export the mesh to the specified format
    mesh.export(output_file)
    print(f"Converted {ply_file} to {output_file}")


def extract_scene_graph(annotations_file: str) -> dict:
    with open(annotations_file) as f:
        annotations = json.load(f)
    scene_id = annotations["sceneId"]

    scene_graph = {}

    for obj in annotations["segGroups"]:
        obj_id = obj["objectId"]
        category = obj["label"]
        centroid = obj["obb"]["centroid"]
        extent = obj["obb"]["axesLengths"]  # TODO: check this
        score = 1.0

        scene_graph[obj_id] = {
            "category": category,
            "centroid": centroid,
            "extent": extent,
            "score": score,
        }

    return scene_id, scene_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for the model")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the scene and annotations file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to the output file",
    )
    parser.add_argument(
        "--bright_factor",
        type=float,
        default=1.5,
        help="Factor to adjust the brightness of the colors",
    )

    args = parser.parse_args()

    output_annotations_file = os.path.join(args.output, "scannetpp_ground_truth_scene_graph.json")
    if os.path.exists(output_annotations_file):
        with open(output_annotations_file, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {}

    annotations_file = os.path.join(args.input, "segments_anno.json")
    scene_id, scene_graph = extract_scene_graph(annotations_file)

    annotations[scene_id] = scene_graph
    with open(output_annotations_file, "w") as f:
        json.dump(annotations, f, indent=4)

    print(f"Scene ID: {scene_id}")
    print(f"Saved scene graph annotations to {output_annotations_file}")

    scene_file = os.path.join(args.input, "mesh_aligned_0.05.ply")
    output_file = os.path.join(args.output, scene_id, f"{scene_id}.obj")
    os.makedirs(os.path.join(args.output, scene_id), exist_ok=True)
    convert_ply_to_format(scene_file, output_file, args.bright_factor)
    print(f"Saved obj file to {output_file}")
