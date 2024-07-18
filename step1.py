from typing import List, Tuple

import click
import copick
import numpy as np
import zarr

import deepfinder.utils.copick_tools as tools
from deepfinder.utils.target_build import TargetBuilder

@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command()
@click.option("--config", type=str, required=True, help="Path to the configuration file.")
@click.option(
    "--target",
    type=(str, str, str, int),
    required=False,
    help="Tuples of object name, user id, session id and radius.",
    multiple=True,
)
@click.option(
    "--seg-target",
    type=(str, str, str),
    required=False,
    help="Tuples of object name, user id and session id for segmentation.",
    multiple=True,
)
@click.option(
    "--tomo-ids",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="Comma separated list of Tomogram IDs.",
)
@click.option("--voxel-size", type=float, default=10, help="Voxel size.")
@click.option("--tomogram-algorithm", type=str, default="wbp", help="Tomogram algorithm.")
@click.option("--out-name", type=str, default="spheretargets", help="Target name.")
@click.option("--out-user-id", type=str, default="train-deepfinder", help="User ID for output.")
@click.option("--out-session-id", type=str, default="0", help="Session ID for output.")
def create_local(
    config: str,
    target: List[Tuple[str, str, str, int]],
    seg_target: List[Tuple[str, str, str]],
    tomo_ids: str,
    voxel_size: float = 10,
    tomogram_algorithm: str = "wbp",
    out_name: str = "spheretargets",
    out_user_id: str = "train-deepfinder",
    out_session_id: str = "0",
):

    # Load CoPick root
    copickRoot = copick.from_file(config)

    train_targets = {}
    for t in target:
        info = {
            "label": copickRoot.get_object(t[0]).label,
            "user_id": t[1],
            "session_id": t[2],
            "radius": t[3],
            "is_particle_target": True,
        }
        train_targets[t[0]] = info

    for t in seg_target:
        info = {
            "label": copickRoot.get_object(t[0]).label,
            "user_id": t[1],
            "session_id": t[2],
            "radius": None,
            "is_particle_target": False,
        }
        train_targets[t[0]] = info

    process_create_command(config, train_targets, tomo_ids, voxel_size, tomogram_algorithm, out_name, out_user_id, out_session_id, is_dataportal=False)


@cli.command()
@click.option("--config", type=str, required=True, help="Path to the configuration file.")
@click.option(
    "--target",
    type=(str, int),
    required=False,
    help="Tuples of object name and radius (in voxels).",
    multiple=True,
)
@click.option(
    "--seg-target",
    type=(str),
    required=False,
    help="Object name for segmentation.",
    multiple=True,
)
@click.option(
    "--tomo-ids",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="Comma separated list of Tomogram IDs.",
)
@click.option("--voxel-size", type=float, default=10, help="Voxel size.")
@click.option("--tomogram-algorithm", type=str, default="wbp", help="Tomogram algorithm.")
@click.option("--out-name", type=str, default="spheretargets", help="Target name.")
@click.option("--out-user-id", type=str, default="train-deepfinder", help="User ID for output.")
@click.option("--out-session-id", type=str, default="0", help="Session ID for output.")
def create_from_dataportal(
    config: str,
    target: List[Tuple[str, int]],
    seg_target: List[ str ],
    tomo_ids: str,
    voxel_size: float = 10,
    tomogram_algorithm: str = "wbp",
    out_name: str = "spheretargets",
    out_user_id: str = "train-deepfinder",
    out_session_id: str = "0",
):
    # Load CoPick root
    copickRoot = copick.from_file(config)

    train_targets = {}
    for t in target:
        info = {
            "label": copickRoot.get_object(t[0]).label,
            "radius": t[1],
            "is_particle_target": True,
        }
        train_targets[t[0]] = info

    for t in seg_target:
        info = {
            "label": copickRoot.get_object(t).label,
            "radius": None,
            "is_particle_target": False,
        }
        train_targets[t] = info

    # Additional logic to source data from the data portal can be added here
    process_create_command(config, train_targets, tomo_ids, voxel_size, tomogram_algorithm, out_name, out_user_id, out_session_id, is_dataportal=True)


def process_create_command(
    config: str,
    train_targets: dict,
    tomo_ids: str,
    voxel_size: float,
    tomogram_algorithm: str,
    out_name: str,
    out_user_id: str,
    out_session_id: str,
    is_dataportal: bool,
):
    # Load CoPick root
    copickRoot = copick.from_file(config)

    target_names = list(train_targets.keys())

    # Radius list
    max_target = max(e["label"] for e in train_targets.values())
    radius_list = np.zeros((max_target,), dtype=np.uint8)

    for _key, value in train_targets.items():
        radius_list[value["label"] - 1] = value["radius"] if value["radius"] is not None else 0

    # Load tomo_ids
    tomo_ids = [run.name for run in copickRoot.runs] if tomo_ids is None else tomo_ids.split(",")

    # Add Spherical Targets to Mebranes
    tbuild = TargetBuilder()

    # Create Empty Target Volume
    target_vol = tools.get_target_empty_tomogram(copickRoot, voxelSize=voxel_size, tomoAlgorithm=tomogram_algorithm)

    # Iterate Through All Runs
    for tomoID in tomo_ids:
        # Extract TomoID and Associated Run
        copickRun = copickRoot.get_run(tomoID)

        # Reset Target As Empty Array
        # (If Membranes or Organelle Segmentations are Available, add that As Well)
        target_vol[:] = 0

        # Applicable segmentations
        query_seg = []
        for target_name in target_names:
            if not train_targets[target_name]["is_particle_target"]:
                if is_dataportal:
                    query_seg += copickRun.get_segmentations(
                        name=target_name,
                        voxel_size=voxel_size,
                        is_multilabel=False,
                    )
                else:
                    query_seg += copickRun.get_segmentations(
                        name=target_name,
                        user_id=train_targets[target_name]["user_id"],
                        session_id=train_targets[target_name]["session_id"],
                        voxel_size=voxel_size,
                        is_multilabel=False,
                    )

        # Add Segmentations to Target
        for seg in query_seg:
            classLabel = copickRoot.get_object(seg.name).label
            segvol = zarr.open(seg.zarr())["0"]

            target_vol[:] = np.array(segvol) * classLabel

        # Applicable picks
        query = []
        for target_name in target_names:
            if train_targets[target_name]["is_particle_target"]:
                if is_dataportal:
                    query += copickRun.get_picks(
                        object_name=target_name
                    )
                else:
                    query += copickRun.get_picks(
                        object_name=target_name,
                        user_id=train_targets[target_name]["user_id"],
                        session_id=train_targets[target_name]["session_id"],
                    )

        # Read Particle Coordinates and Write as Segmentation
        objl_coords = []
        for picks in query:
            classLabel = copickRoot.get_object(picks.pickable_object_name).label

            if classLabel is None:
                print("Missing Protein Label: ", picks.pickable_object_name)
                exit(-1)

            for ii in range(len(picks.points)):
                objl_coords.append(
                    {
                        "label": classLabel,
                        "x": picks.points[ii].location.x / voxel_size,
                        "y": picks.points[ii].location.y / voxel_size,
                        "z": picks.points[ii].location.z / voxel_size,
                        "phi": 0,
                        "psi": 0,
                        "the": 0,
                    },
                )

        # Create Target For the Given Coordinates and Sphere Diameters
        target = tbuild.generate_with_spheres(objl_coords, target_vol, radius_list).astype(np.uint8)

        # Write the Target Tomogram as OME Zarr
        tools.write_ome_zarr_segmentation(copickRun, target, voxel_size, out_name, out_user_id, out_session_id)


if __name__ == "__main__":
    cli()
