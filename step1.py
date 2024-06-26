from deepfinder.utils.target_build import TargetBuilder
import copick
import deepfinder.utils.copick_tools as tools
import numpy as np
import os
import click

@click.group()
@click.pass_context
def cli(ctx):
    pass

@cli.command()
@click.option("--config", type=str, required=True, help="Path to the configuration file.")
@click.option("--radius_list", type=str, required=True, help="Comma separated sizes.")
@click.option("--tomoIDs", type=str, required=True, help="Comma separated list of Tomogram IDs.")
@click.option("--voxelSize", type=float, default=10, help="Voxel size.")
@click.option("--targetName", type=str, default="spheretargets", help="Target name.")
@click.option("--userID", type=str, default="train-deepfinder", help="User ID.")
def create(
    config: str,
    radius_list: str,
    tomoIDs: str,
    voxelSize: float = 10,
    targetName: str = 'spheretargets',
    userID: str = 'train-deepfinder',
):
    # Load CoPick root
    copickRoot = copick.from_file(config)

    # List for How Large the Target Sizes should be
    radius_list = [int(x) for x in radius_list.split(',')]

    # Load TomoIDs
    tomoIDs = tomoIDs.split(',')

    # Add Spherical Targets to Mebranes
    tbuild = TargetBuilder()

    # Create Empty Target Volume
    target_vol = tools.get_target_empty_tomogram(copickRoot)

    # Iterate Through All Runs
    for tomoInd in range(len(tomoIDs)):

        # Extract TomoID and Associated Run
        tomoID = tomoIDs[tomoInd]
        copickRun = copickRoot.get_run(tomoID)

        # Read Particle Coordinates and Write as Segmentation
        objl_coords = []
        for proteinInd in range(len(copickRun.picks)):
            picks = copickRun.picks[proteinInd]
            classLabel = tools.get_pickable_object_label(copickRoot, picks.pickable_object_name)

            if classLabel == None:
                print('Missing Protein Label: ', picks.pickable_object_name)
                exit()

            for ii in range(len(picks.points)):
                objl_coords.append({'label': classLabel,
                                    'x': picks.points[ii].location.x / voxelSize,
                                    'y': picks.points[ii].location.y / voxelSize,
                                    'z': picks.points[ii].location.z / voxelSize,
                                    'phi': 0, 'psi': 0, 'the': 0})

        # Reset Target As Empty Array
        # (If Membranes or Organelle Segmentations are Available, add that As Well)
        target_vol[:] = 0

        # Create Target For the Given Coordinates and Sphere Diameters
        target = tbuild.generate_with_spheres(objl_coords, target_vol, radius_list).astype(np.uint8)

        # Write the Target Tomogram as OME Zarr
        tools.write_ome_zarr_segmentation(copickRun, target, voxelSize, targetName, userID)

if __name__ == "__main__":
    cli()