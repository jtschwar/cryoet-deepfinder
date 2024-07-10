from copick.impl.filesystem import CopickRootFSSpec
import deepfinder.utils.copick_tools as tools
import my_polnet_utils as utils
import scipy.ndimage as ndimage
from tqdm import tqdm 
import numpy as np
import click, os

@click.group()
@click.pass_context
def cli(ctx):
    pass

####################################################################

proteins = {'apo': {'name': 'apo-ferritin', 'diameter': 130}, 
            'ribo80S': {'name': 'ribosome', 'diameter': 310}}

@cli.command()
@click.option(    
    "--predict-config",
    type=str,
    required=True,
    help="Path to the copick config file.",)
@click.option(
    "--nclass",
    type=int,
    required=True,
    help="Number of classes.",
)
@click.option(
    "--user-id", 
    type=str, 
    default='deepfinder', 
    show_default=True, 
    help="User ID filter for input."
)
@click.option(
    "--segmentation-session-id", 
    type=str, 
    default=None, 
    show_default=True, 
    help="Session ID filter for input."
)
@click.option(
    "--coordinates-session-id", 
    type=str, 
    default=None, 
    show_default=True, 
    help="Session ID filter for input."
)
@click.option(
    "--voxel-size",
    type=float,
    required=False,
    default=10.0,
    show_default=True,
    help="Voxel size of the tomograms to segment.",
)
@click.option(
    "--copick-write-directory",
    type=str, 
    required=True,
    default="copick_project/ExperimentRuns",
    show_default=True,
    help="Write STAR Files in Specified Directory.",
)
@click.option(
    "--relion-write-directory/--no-relion-write-directory",
    type=str, 
    default=None, 
    help="Write STAR Files in Specified Directory.",
)
@click.option(
    "--parallel-mpi/--no-parallel-mpi",
    default=False,
    help="Run Parallel Localization Inference with MPI.",
)
def localize(
    predict_config: str, 
    nclass: int, 
    user_id: str, 
    segmentation_session_id: str, 
    coordinates_session_id: str, 
    copick_write_directory: str ,     
    voxel_size: float = 10,
    relion_write_directory: str = None,
    parallel_mpi: bool = False,         
    ):

    # Determine if Using MPI or Sequential Processing
    if parallel_mpi: 
        # Initialize MPI (Get Rank and nProc)
        from mpi4py import MPI
        comm = MPI.COMM_WORLD; nProcess = comm.Get_size(); rank = comm.Get_rank()
    else:
        nProcess = 1; rank = 0

    ############## (Step 1) Initialize segmentation task: ##############

    tags = list(proteins)            

    # Load CoPick root
    copickRoot = CopickRootFSSpec.from_file(predict_config)

    # Load Evaluate TomoIDs
    evalTomos = [run.name for run in copickRoot.runs]

    # Create Temporary Empty Folder 
    for tomoInd in range(len(evalTomos)): 

        if (tomoInd + 1) % nProcess == rank:

            tomoID = evalTomos[tomoInd]
            copickRun = copickRoot.get_run(tomoID)
            labelmap = tools.get_copick_segmentation(copickRun, segmentation_session_id, user_id)[0][:]

            for label in range(2, nclass):

                    proteinName = proteins[tags[label-2]]['name']
                    print(f'Finding Predictions for : {proteinName}')
                    label_objs, num_features = ndimage.label(labelmap == label)

                    # Estimate Coordiantes from CoM for LabelMaps
                    deepFinderCoords = []
                    for object_num in tqdm(range(1, num_features+1)):
                        com = ndimage.center_of_mass(label_objs == object_num)
                        swapped_com = (com[2], com[1], com[0])
                        deepFinderCoords.append(swapped_com)
                    deepFinderCoords = np.array(deepFinderCoords)

                    # Estimate Distance Threshold Based on 1/2 of Particle Diameter
                    threshold = np.ceil(  proteins[tags[label-2]]['diameter'] / (voxel_size * 3) )

                    try: 
                        # Remove Double Counted Coordinates
                        deepFinderCoords = utils.remove_repeated_picks(deepFinderCoords, threshold)

                        # Convert from Voxel to Physical Units
                        deepFinderCoords *= voxel_size

                        # Append Euler Angles to Coordinates [ Expand Dimensions from Nx3 -> Nx6 ]
                        deepFinderCoords = np.concatenate((deepFinderCoords, np.zeros(deepFinderCoords.shape)),axis=1)

                        # # Write the Starfile for Visualization
                        if relion_write_directory is not None:
                            os.makedirs( os.path.join(relion_write_directory,tomoID), exist_ok=True)
                            tools.write_relion_output(proteinName, None, deepFinderCoords,
                                                      os.path.join(relion_write_directory,tomoID), pixelSize=1) 

                    except Exception as e:
                        print(f"Error processing label {proteinName} in tomo {tomoID}: {e}")
                        deepFinderCoords = np.array([]).reshape(0,6)

                    # Save Picks in Copick Format / Directory 
                    tools.write_copick_output(proteinName, tomoID, deepFinderCoords, copick_write_directory, 
                                              pickMethod=user_id, sessionID = coordinates_session_id)

    print('Extraction of Particle Coordinates Complete!')

if __name__ == "__main__":
    cli()
