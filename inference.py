import argparse
import json
import logging
import os
import re
from copy import deepcopy

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import trimesh.transformations as tf
from bs4 import BeautifulSoup


# The following line sets the root logger level as well.
# It's equivalent to both previous statements combined:
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from model import load_model_and_dataloader, get_model_response

# Load model and tokenizer once at the start
model_path = "checkpoints/merged_weights_grounded_obj_ref"
model_base = None
load_8bit = False
load_4bit = False
load_bf16 = True
max_new_tokens = 5000
obj_context_feature_type = "text"


def import_data(data_path: str, annotations_path: str, single_object_only: bool):
    # load prompts
    with open(data_path, "r") as f:
        prompts = json.load(f)
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    # filters
    if single_object_only:
        prompts = prompts[prompts["target_type"] == "single"]

    # add
    # def add_annotations(row):
    #     obj = annotations[row.scene_id][str(row.target_id)]
    #     return pd.Series([obj["centroid"], obj["extent"]])

    # prompts[["target_centroid", "target_extent"]] = prompts[["scene_id", "target_id"]].apply(add_annotations, axis=1)
    return prompts


def create_bbox(center, extents, color=[1, 0, 0], radius=0.02):
    """Create a colored bounding box with given center, extents, and line thickness."""
    # ... [The same code as before to define corners and lines] ...
    print(extents)
    print(type(extents))
    extents = extents.replace("[", "").replace("]", "")
    center = center.replace("[", "").replace("]", "")
    extents = [float(x.strip()) for x in extents.split(",")]
    center = [float(x.strip()) for x in center.split(",")]
    angle = -np.pi / 2  # 90 degrees
    axis = [1, 0, 0]  # Rotate around x-axis
    R = tf.rotation_matrix(angle, axis)
    center_homogeneous = np.append(center, 1)
    extents_homogeneous = np.append(extents, 1)

    # Apply the rotation to the center and extents
    rotated_center = np.dot(R, center_homogeneous)[:3]
    rotated_extents = np.dot(R, extents_homogeneous)[:3]

    sx, sy, sz = rotated_extents
    x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
    y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
    z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0, :] = corners_3d[0, :] + float(rotated_center[0])
    corners_3d[1, :] = corners_3d[1, :] + float(rotated_center[1])
    corners_3d[2, :] = corners_3d[2, :] + float(rotated_center[2])
    corners_3d = np.transpose(corners_3d)

    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    cylinders = []
    for line in lines:
        p0, p1 = corners_3d[line[0]], corners_3d[line[1]]
        cylinders.append(create_cylinder_mesh(p0, p1, color, radius))
    return cylinders


def highlight_clusters_in_mesh(
    centroids_extents_detailed,
    centroids_extends_refer,
    mesh,
    output_dir,
    output_file_name="highlighted_mesh.obj",
):
    print("*" * 50)
    # Visualize the highlighted points by drawing 3D bounding boxes overlay on a mesh
    old_mesh = deepcopy(mesh)
    output_path = os.path.join(output_dir, "mesh_vis")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create a combined mesh to hold both the original and the bounding boxes
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh += old_mesh

    # Draw bounding boxes for each centroid and extent
    for center, extent in centroids_extents_detailed:
        print("center: ", center)
        print("extent: ", extent)
        bbox = create_bbox(center, extent, color=[1, 1, 0])  # yellow color for all boxes
        for b in bbox:
            combined_mesh += b

    for center, extent in centroids_extends_refer:
        bbox = create_bbox(center, extent, color=[0, 1, 0])
        for b in bbox:
            combined_mesh += b

    # Save the combined mesh
    output_file_path = os.path.join(output_path, output_file_name)
    o3d.io.write_triangle_mesh(output_file_path, combined_mesh, write_vertex_colors=True)
    print("*" * 50)
    return output_file_path


def extract_objects(text):
    return re.findall(r"<obj_\d+>", text)


# Parse the scene graph into a dictionary
def parse_scene_graph(scene_graph):
    scene_dict = {}
    matches = re.findall(r"<obj_(\d+)>: (\{.*?\})", scene_graph)
    for match in matches:
        obj_id = f"<obj_{match[0]}>"
        obj_data = eval(match[1])
        scene_dict[obj_id] = obj_data
    return scene_dict


def get_centroids_extents(obj_list, scene_dict):
    centroids_extents = []
    for obj in obj_list:
        if obj in scene_dict:
            # FIXME: need to be careful about using eval here
            centroid = eval(scene_dict[obj]["centroid"])
            extent = eval(scene_dict[obj]["extent"])
            centroids_extents.append((centroid, extent))
    return centroids_extents


def get_chatbot_response(user_chat_input, scene_id, temperature=0.2, top_p=0.9, max_new_tokens=50):
    # Get the response from the model
    prompt, response = get_model_response(
        model=model,
        tokenizer=tokenizer,
        data_loader=data_loader,
        scene_id=scene_id,
        user_input=user_chat_input,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return scene_id, prompt, response


def language_model_forward(
    scene_id: str, prompt: str, top_p: float = 0.9, temperature: float = 0.2, max_new_tokens: int = 50
):
    # session_state = Session.create_for_scene(dropdown_scene)
    # session_state.chat_history_for_display.append(
    #     (prompt, None)
    # )  # append in a tuple format, first is user input, second is assistant response

    # yield session_state, None, session_state.chat_history_for_display

    # Load in a 3D model
    # file_name = f"{scene_id}.obj"
    # original_model_path = os.path.join(GRAND3D_Settings.data_path, scene_id, file_name)
    # print("original_model_path: ", original_model_path)

    # get chatbot response
    print(f"Scene ID: {scene_id}")
    print(f"Model Prompt: {prompt}")
    try:
        scene_id, scene_graph, response = get_chatbot_response(
            prompt, scene_id, top_p=top_p, temperature=temperature, max_new_tokens=max_new_tokens
        )
    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"Out of memory error: {e}")
        return None

    # use scene_graph and response to get centroids and extents
    # Parse the scene graph into a dictionary
    scene_dict = parse_scene_graph(scene_graph)
    # print("Model Input: " + str(scene_dict))
    # print("Model Response: " + response)

    # Parse the response to get detailed and refer expression groundings
    soup = BeautifulSoup(response, "html.parser")
    # detailed_grounding_html = str(soup.find("detailed_grounding"))
    refer_expression_grounding_html = str(soup.find("refer_expression_grounding"))

    # Extract objects from both sections
    # detailed_objects = extract_objects(detailed_grounding_html)
    refer_objects = extract_objects(refer_expression_grounding_html)

    # Extract objects from both sections
    # print("detailed_objects: ", detailed_objects)
    print("refer_objects: ", refer_objects)

    # Perform set subtraction to get remaining objects
    # remaining_objects = list(set(detailed_objects) - set(refer_objects))
    # print("remaining_objects: ", remaining_objects)

    # centroids_extents_detailed = get_centroids_extents(remaining_objects, scene_dict)
    # print("centroids_extents_detailed: ", centroids_extents_detailed)
    centroids_extents_refer = get_centroids_extents(refer_objects, scene_dict)
    print("centroids_extents_refer: ", centroids_extents_refer)
    print("=" * 50)

    return centroids_extents_refer, refer_objects


class AccuracyAtIoU:
    def __init__(self, iou) -> None:
        self.iou_threshold = iou

        # internal state
        self.reset()

    def reset(self):
        self.num_samples = 0
        self.num_correct = 0

    @staticmethod
    def _compute_iou(
        centroid_1: list[float], extent_1: list[float], centroid_2: list[float], extent_2: list[float]
    ) -> float:
        box_min_1 = np.array([c - e / 2 for c, e in zip(centroid_1, extent_1)])
        box_max_1 = np.array([c + e / 2 for c, e in zip(centroid_1, extent_1)])
        box_min_2 = np.array([c - e / 2 for c, e in zip(centroid_2, extent_2)])
        box_max_2 = np.array([c + e / 2 for c, e in zip(centroid_2, extent_2)])

        intersect = min(
            np.prod(np.clip(box_max_2 - box_min_1, a_min=0.0, a_max=None)),
            np.prod(np.clip(box_max_1 - box_min_2, a_min=0.0, a_max=None)),
        )
        union = np.prod(box_max_1 - box_min_1) + np.prod(box_max_2 - box_min_2) - intersect

        return intersect / union if union > 0 else 0.0

    def __call__(self, output, target):
        assert len(output) <= 1, "multi-box outputs or GT not supported"
        if target:
            self.num_samples += 1
            iou = self._compute_iou(output[0][0], output[0][1], target[0], target[1]) if output else 0.0
            if iou > self.iou_threshold:
                self.num_correct += 1
                return True
            return False
        else:
            return len(output) == 0

    def get(self):
        return self.num_correct / self.num_samples if self.num_samples else np.nan

    def __str__(self) -> str:
        return f"AccuracyAtIoU{self.iou_threshold}"


def visualize(original_model_path, centroids_extents_refer, output_dir):
    # Load the GLB mesh
    mesh = o3d.io.read_triangle_mesh(original_model_path)

    highlighted_model_path = highlight_clusters_in_mesh(
        [],
        centroids_extents_refer,
        mesh,
        output_dir,
        output_file_name="highlighted_model.obj",
    )
    return highlighted_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", "-i", type=str, dest="prompts", default="data/scannetpp/test_prompts.csv")
    parser.add_argument("--scenes", type=str, dest="scenes", default="data/scannetpp")
    parser.add_argument(
        "--object-mapping",
        type=str,
        dest="scene_to_obj_mapping",
        default="data/scannetpp/scannetpp_ground_truth_scene_graph.json",
    )
    parser.add_argument("--top-p", "-p", type=float, dest="top_p", default=0.9)
    parser.add_argument("--temperature", "-t", type=float, dest="temp", default=0.9)
    parser.add_argument("--max-new-tokens", "-m", type=int, dest="max_new_tokens", default=5000)
    parser.add_argument("--visualize", "-v", action="store_true", dest="visualize")
    parser.add_argument("--output", "-o", type=str, dest="output_dir", default="data/scannetpp/outputs")
    parser.add_argument("--single-object-only", action="store_true", dest="single_object_only")
    args = parser.parse_args()

    inp_dir = "data/scannetpp"
    # metrics = [AccuracyAtIoU(0.25), AccuracyAtIoU(0.50)]
    outputs = []

    tokenizer, model, data_loader = load_model_and_dataloader(
        model_path=model_path,
        model_base=model_base,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        load_bf16=load_bf16,
        scene_to_obj_mapping=args.scene_to_obj_mapping,
        device_map="cpu",
    )  # Huggingface Zero-GPU has to use .to(device) to set the device, otherwise it will fail

    model.to("cuda")  # Huggingface Zero-GPU requires explicit device placement

    # load prompts
    prompts = import_data(args.prompts, args.scene_to_obj_mapping, single_object_only=args.single_object_only)

    for sample in prompts["grounding"]:
        torch.cuda.empty_cache()
        # generate prediction
        response = language_model_forward(
            sample["scene_id"],
            sample["text"],
            top_p=args.top_p,
            temperature=args.temp,
            max_new_tokens=args.max_new_tokens,
        )
        if response is not None:
            centroids_extents_refer, refer_ids = response

            # log output
            outputs.append(
                {
                    "prompt_id": sample["id"],
                    "scene_id": sample["scene_id"],
                    "prompt": sample["text"],
                    "predicted_ids": refer_ids,
                    "predicted_boxes": centroids_extents_refer,
                    # "ground_truth": {
                    #     "ids": sample.target_id,
                    #     "label": sample.target_label,
                    #     "boxes": [sample.target_centroid, sample.target_extent],
                    # },
                }
            )

            # evaluate
            # for m in metrics:
            #     m(centroids_extents_refer, [sample.target_centroid, sample.target_extent])

            # visualize
            if args.visualize:
                original_model_path = os.path.join(args.scenes, sample["scene_id"], f"{sample['scene_id']}.obj")
                model_path = visualize(original_model_path, centroids_extents_refer, args.output_dir)

    # save outputs
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(outputs, f, indent=2)

    # print("Final Metrics:")
    # print("--------------")
    # for m in metrics:
    #     print(f"{m} = {m.get()}")
