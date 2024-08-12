import deepfinder.utils.copick_tools as tools
import deepfinder.utils.eval as eval
import scipy.ndimage as ndimage
import click, copick, os, json
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
    "--segmentation-name",
    type=str,
    required=False,
    default="segmentation",
    show_default=True,
    help="Name for segmentation prediction.",    
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
@click.option(
    "--min-protein-size",
    type=float,
    required=False,
    default=0.8,
    help="Specifies the minimum size of protein objects to be considered during the localization process. "
          "This parameter helps filter out small false positives objects based on their size. The value should be between (0, 1.0], "
          "representing a fraction of the typical protein size defined in the configuration. Objects smaller than this threshold will be ignored. ",
)
def localize(
    predict_config: str,
    n_class: int,
    user_id: str, 
    picks_session_id: str,
    segmentation_session_id: str,    
    segmentation_name: str,
    voxel_size: float,
    parallel_mpi: bool = False,
    tomo_ids: str = None,
    path_output: str = "deefinder_predictions/ExperimentRuns",
    starfile_write_path: str = None,
    min_protein_size: float = 0.8, 
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

    # Open and read the Config File
    with open(predict_config, 'r') as file:
        data = json.load(file)
    
    # Get Path Output from Config File
    # path_output = os.path.join(data['overlay_root'], 'ExperimentRuns')

    # Create dictionary with name as key and diameter as value
    proteins = {obj['name']: (obj['radius'], obj['label']) for obj in data['pickable_objects']}

    # Create a reverse dictionary with label as key and name as value
    label_to_name_dict = {obj['label']: obj['name'] for obj in data['pickable_objects']}          

    # Load CoPick root
    copickRoot = copick.from_file(predict_config)

    # Load Evaluate TomoIDs
    evalTomos = tomo_ids.split(",") if tomo_ids is not None else [run.name for run in copickRoot.runs]

    # Create Temporary Empty Folder 
    for tomoInd in tqdm(range(len(evalTomos))):
        if (tomoInd + 1) % nProcess == rank: 
            # Extract TomoID and Associated Run    
            tomoID = evalTomos[tomoInd]
            print(f'Processing Run: {tomoID}')

            copickRun = copickRoot.get_run(tomoID)           
            labelmap = tools.get_copick_segmentation(copickRun, segmentation_name, user_id, segmentation_session_id)[:]
            for label in range(2, n_class):
                
                    protein_name = label_to_name_dict.get(label)
                    print('Finding Predictions for : ', protein_name)
                    label_objs, _ = ndimage.label(labelmap == label)

                    # Filter Candidates based on Object Size
                    # Get the sizes of all objects
                    object_sizes = np.bincount(label_objs.flat)

                    # Filter the objects based on size
                    min_object_size = 4/3 * np.pi * ((proteins[protein_name][0]/voxel_size)**2) * min_protein_size
                    valid_objects = np.where(object_sizes > min_object_size)[0]                          

                    # Estimate Coordiantes from CoM for LabelMaps
                    deepFinderCoords = []
                    for object_num in tqdm(valid_objects):
                        com = ndimage.center_of_mass(label_objs == object_num)
                        swapped_com = (com[2], com[1], com[0])
                        deepFinderCoords.append(swapped_com)
                    deepFinderCoords = np.array(deepFinderCoords)   

                    # Estimate Distance Threshold Based on 1/2 of Particle Diameter
                    threshold = np.ceil(  proteins[protein_name][0] / (voxel_size * 3) )

                    try: 
                        # Remove Double Counted Coordinates
                        deepFinderCoords = eval.remove_repeated_picks(deepFinderCoords, threshold)

                        # Append Euler Angles to Coordinates [ Expand Dimensions from Nx3 -> Nx6 ]
                        deepFinderCoords = np.concatenate((deepFinderCoords, np.zeros(deepFinderCoords.shape)),axis=1)

                        # Write the Starfile for Visualization
                        if starfile_write_path is not None:
                            tomoIDstarfilePath = os.path.join(starfile_write_path,tomoID)
                            os.makedirs(tomoIDstarfilePath, exist_ok=True)
                            tools.write_relion_output(protein_name, None, deepFinderCoords, tomoIDstarfilePath , pixelSize=1) 

                        # Convert from Voxel to Physical Units
                        deepFinderCoords *= voxel_size

                    except Exception as e:
                        print(f"Error processing label {label} in tomo {tomoID}: {e}")
                        deepFinderCoords = np.array([]).reshape(0,6)

                    # Save Picks in Copick Format / Directory 
                    tools.write_copick_output(protein_name, tomoID, deepFinderCoords, path_output, pickMethod=user_id, sessionID = picks_session_id)

    print('Extraction of Particle Coordinates Complete!')

if __name__ == "__main__":
    cli()