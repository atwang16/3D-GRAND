import argparse
import json
import os

import plyfile
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


def extract_instance_annotations(segments_data, annotations):
    segments = np.array(segments_data["segIndices"])
    scene_graph = {}

    for obj in annotations["segGroups"]:
        label = "_".join(obj["label"].split())
        segments_obj = obj["segments"]

        # vertices
        v_mask = np.isin(segments, segments_obj)

        scene_graph[obj["objectId"]] = {
            "category": label,
            "centroid": obj["obb"]["centroid"],
            "extent": obj["obb"]["axesLengths"],
            "score": 1.0,
        }
    return scene_graph


def extract_scene_graph(scene_id: str, scene_dir: str, annotation_dir: str) -> dict:
    # load inst
    segments = json.load(open(os.path.join(scene_dir, "scans", scene_id, f"{scene_id}_vh_clean_2.0.010000.segs.json")))
    annotations = json.load(open(os.path.join(scene_dir, "scans", scene_id, f"{scene_id}_vh_clean.aggregation.json")))
    annotations_obb = json.load(open(os.path.join(annotation_dir, scene_id, f"{scene_id}.semseg.json")))

    for idx, obj_obb in enumerate(annotations_obb["segGroups"]):
        annotations["segGroups"][idx]["obb"] = obj_obb["obb"]

    scene_graph = extract_instance_annotations(segments, annotations)

    return scene_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for the model")
    parser.add_argument("--scene-id", "-i", type=str, required=True, help="ID of scene")
    parser.add_argument(
        "--annotation-dir",
        "-a",
        type=str,
        required=True,
        help="Path to the annotations folder",
    )
    parser.add_argument(
        "--scene-dir",
        "-s",
        type=str,
        required=True,
        help="Path to the scene folder",
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

    output_annotations_file = os.path.join(args.output, "scannet_ground_truth_scene_graph.json")
    if os.path.exists(output_annotations_file):
        with open(output_annotations_file, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {}

    scene_graph = extract_scene_graph(args.scene_id, args.scene_dir, args.annotation_dir)

    annotations[args.scene_id] = scene_graph
    with open(output_annotations_file, "w") as f:
        json.dump(annotations, f, indent=4)

    print(f"Scene ID: {args.scene_id}")
    print(f"Saved scene graph annotations to {output_annotations_file}")

    scene_file = os.path.join(args.scene_dir, "scans", args.scene_id, f"{args.scene_id}_vh_clean_2.ply")
    output_file = os.path.join(args.output, args.scene_id, f"{args.scene_id}.obj")
    os.makedirs(os.path.join(args.output, args.scene_id), exist_ok=True)
    convert_ply_to_format(scene_file, output_file, args.bright_factor)
    print(f"Saved obj file to {output_file}")
