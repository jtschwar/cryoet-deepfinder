import deepfinder.utils.copick_tools as tools
import deepfinder.utils.eval as eval
import scipy.ndimage as ndimage
import click, copick, os
from tqdm import tqdm 
import numpy as np

@click.group()
@click.pass_context
def cli(ctx):
    pass

@cli.command()
@click.option(
    "--predict-config",
    type=str,
    required=True,
    help="Path to the copick config file.",
)
@click.option(
    "--n-class",
    type=int,
    required=True,
    help="Number of classes.",
)
@click.option(
    "--user-id", 
    type=str, 
    default="deepfinder", 
    show_default=True, 
    help="User ID filter for input."
)
@click.option(
    "--picks-session-id", 
    type=str, 
    default=None, 
    show_default=True, 
    help="Session ID filter for input."
)
@click.option(
    "--segmentation-session-id", 
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
    "--tomogram-algorithm",
    type=str,
    required=False,
    default="wbp",
    show_default=True,
    help="Tomogram Algorithm.",
)
@click.option(
    "--parallel-mpi/--no-parallel-mpi",
    default=False,
    help="Patch of Volume for Input to Network.",
)
@click.option(
    "--tomo-ids",
    type=str,
    required=False,
    default=None,
    help="Tomogram IDs to Segment.",
)
@click.option(
    "--path-output",
    type=str,
    required=False,
    default="deefinder_predictions/ExperimentRuns",
    help="Path to Copick Project to Write Results"
)
@click.option(
    "--starfile-write-path",
    type=str,
    required=False,
    default=None,
    help="Write Path to Save Starfile Coordinate Files Per Protein.",
)
def localize(
    predict_config: str,
    n_class: int,
    user_id: str, 
    picks_session_id: str,
    segmentation_session_id: str,    
    voxel_size: float,
    parallel_mpi: bool = False,
    tomo_ids: str = None,
    path_output: str = "deefinder_predictions/ExperimentRuns",
    starfile_write_path: str = None
    ):

    # Determine if Using MPI or Sequential Processing
    if parallel_mpi:
        from mpi4py import MPI

        # Initialize MPI (Get Rank and nProc)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nProcess = comm.Get_size()
    else:
        nProcess = 1
        rank = 0

    # TODO: Query Proteins Shapes from Config File
    proteins = {'apo': {'name': 'apo-ferritin', 'diameter': 130}, 
                'ribo80S': {'name': 'ribosome', 'diameter': 310},
                'VLP': {'name': 'virus-like-particle', 'diameter': 285} }

    tags = list(proteins)            

    # Load CoPick root
    copickRoot = copick.from_file(predict_config)

    # Load Evaluate TomoIDs
    evalTomos = tomo_ids.split(",") if tomo_ids is not None else [run.name for run in copickRoot.runs]

    # Create Temporary Empty Folder 
    for tomoInd in tqdm(range(len(evalTomos))):
        if (tomoInd + 1) % nProcess == rank:     
            tomoID = evalTomos[tomoInd]
            copickRun = copickRoot.get_run(tomoID)
            labelmap = tools.get_copick_segmentation(copickRun, segmentation_session_id, user_id)[0][:]

            for label in range(2, n_class):

                    print('Finding Predictions for : ', proteins[tags[label-2]]['name'])
                    label_objs, num_features = ndimage.label(labelmap == label)

                    # Estimate Coordiantes from CoM for LabelMaps
                    deepFinderCoords = []
                    for object_num in range(1, num_features+1):
                        com = ndimage.center_of_mass(label_objs == object_num)
                        swapped_com = (com[2], com[1], com[0])
                        deepFinderCoords.append(swapped_com)
                    deepFinderCoords = np.array(deepFinderCoords)

                    # Estimate Distance Threshold Based on 1/2 of Particle Diameter
                    threshold = np.ceil(  proteins[tags[label-2]]['diameter'] / (voxel_size * 3) )

                    try: 
                        # Remove Double Counted Coordinates
                        deepFinderCoords = eval.remove_repeated_picks(deepFinderCoords, threshold)

                        # Convert from Voxel to Physical Units
                        deepFinderCoords *= voxel_size

                        # Append Euler Angles to Coordinates [ Expand Dimensions from Nx3 -> Nx6 ]
                        deepFinderCoords = np.concatenate((deepFinderCoords, np.zeros(deepFinderCoords.shape)),axis=1)

                        # Write the Starfile for Visualization
                        if starfile_write_path is not None:
                            tomoIDstarfilePath = os.path.join(starfile_write_path,tomoID)
                            os.makedirs(tomoIDstarfilePath, exist_ok=True)
                            tools.write_relion_output(proteins[tags[label-2]]['name'], None, deepFinderCoords, tomoIDstarfilePath , pixelSize=1) 
                    
                    except Exception as e:
                        print(f"Error processing label {label} in tomo {tomoID}: {e}")
                        deepFinderCoords = np.array([]).reshape(0,6)

                    # Save Picks in Copick Format / Directory 
                    tools.write_copick_output(proteins[tags[label-2]]['name'], tomoID, deepFinderCoords, path_output, pickMethod=user_id, sessionID = picks_session_id)

    print('Extraction of Particle Coordinates Complete!')

if __name__ == "__main__":
    cli()