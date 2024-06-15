from scipy.spatial.transform import Rotation as R
import json, zarr, starfile, os
import numpy as np

def read_copick_tomogram_group(copickRoot, voxelSize, tomoAlgorithm, tomoID=None):
    """ Find the Zarr Group Relating to a Copick Tomogram.

    Args:
        copickRoot: Voxel size for the segmentation.
        voxelSize: Name of the segmentation.
        tomoAlgorithm: Session ID for the segmentation.
        tomoID: 

    Returns:
        CopickSegmentation: The newly created segmentation object.
    """

    # Get First Run and Pull out Tomgram
    if tomoID == None:  run = copickRoot.get_run( copickRoot.runs[0].name )
    else:               run = copickRoot.get_run( tomoID )

    tomogram = run.get_voxel_spacing(voxelSize).get_tomogram(tomoAlgorithm)

    # Convert Zarr into Vol and Extract Shape
    group = zarr.open( tomogram.zarr() )

    return group 

def get_copick_tomogram(copickRoot, voxelSize=10, tomoAlgorithm='denoised', tomoID=None):
    """ Return a Tomogram from a Copick Run.

    Args:
        copickRoot: Voxel size for the segmentation.
        voxelSize: Name of the segmentation.
        tomoAlgorithm: Session ID for the segmentation.
        tomoID: 

    Returns:
        CopickSegmentation: The newly created segmentation object.
    """    

    group = read_copick_tomogram_group(copickRoot, voxelSize, tomoAlgorithm, tomoID)

    # Return Volume
    return list(group.arrays())[0][1]

def get_copick_tomogram_shape(copickRoot, voxelSize=10, tomoAlgorithm='denoised'):
    """ Return a Tomogram Dimensions (nx, ny, nz) from a Copick Run. 

    Args:
        copickRoot: Voxel size for the segmentation.
        voxelSize: Name of the segmentation.
        tomoAlgorithm: Session ID for the segmentation.

    Returns:
        CopickSegmentation: The newly created segmentation object.
    """

    # Return Volume Shape
    return get_copick_tomogram(copickRoot, voxelSize, tomoAlgorithm).shape

def get_target_empty_tomogram(copickRoot, voxelSize=10, tomoAlgorithm='denoised'):
    """ Return an Empty Tomogram with Equivalent Dimensions (nx, ny, nz) from a Copick Run.

    Args:
        copickRoot: Voxel size for the segmentation.
        voxelSize: Name of the segmentation.
        tomoAlgorithm: Session ID for the segmentation.
    Returns:
        CopickSegmentation: The newly created segmentation object.
    """

    return np.zeros(get_copick_tomogram_shape(copickRoot, voxelSize, tomoAlgorithm), dtype=np.int8)

def get_ground_truth_coordinates(copickRun, voxelSize, proteinIndex):
    """ Get the Ground Truth Coordinates From Copick and Return as a Numpy Array.

    Args:
        copickRun: Voxel size for the segmentation.
        voxelSize: Name of the segmentation.
        proteinIndex: Session ID for the segmentation.

    Returns:
        coords: The newly created segmentation object.
    """

    picks = copickRun.picks[proteinIndex]

    coords = []
    for ii in range(len(picks.points)):
        coords.append( (picks.points[ii].location.x / voxelSize,
                       picks.points[ii].location.y / voxelSize,
                       picks.points[ii].location.z / voxelSize) )

    return np.array(coords)


# I need to Figure Out if I want Option 1 or Option 2..
# def get_pickable_object_label(copickRun, objectName):
#     for ii in range(len(copickRun.picks)):
#         if copickRun.picks[ii].pickable_object_name == objectName:
#             return ii 

def get_pickable_object_label(copickRoot, objectName):
    for ii in range(len(copickRoot.pickable_objects)):
        if copickRoot.pickable_objects[ii].name == objectName:
            return copickRoot.pickable_objects[ii].label 



def read_copick_json(filePath):

    # Load JSON data from a file
    with open( os.path.join(filePath), 'r') as jFile:
        data = json.load(jFile)

    # Initialize lists to hold the converted data
    coordinates = []; eulerAngles = []

    # Loop through each point in the JSON data
    for point in data['points']:

        rotationMatrix = []

        # Extract the location and convert it to a NumPy array
        currLocation = np.array( [ point['location']['x'], point['location']['y'], point['location']['z'] ] )

        # Extract the transformation matrix and convert it to a NumPy array
        rotationMatrix = R.from_matrix(np.array(point['transformation_'])[:3,:3])
        currEulerAngles = rotationMatrix.as_euler('ZYZ', degrees=True)

        coordinates.append(currLocation)
        eulerAngles.append(currEulerAngles)

    return  np.hstack((np.array(coordinates), np.array(eulerAngles)))

def convert_copick_coordinates_to_xml(copickRun, xml_objects, pixelSize = 10):
    
    picks = copickRun.picks
    for proteinLabel in range(len(picks)):
        xml_objects.append()

    return xml_objects

def write_relion_output(specimen, tomoID, coords, outputDirectory='refinedCoPicks/ExperimentRuns', pixelSize = 10 ):

    outputStarFile = {}

    # Coordinates
    if coords.shape[0] > 0:
        outputStarFile['rlnCoordinateX'] = coords[:,0] / pixelSize
        outputStarFile['rlnCoordinateY'] = coords[:,1] / pixelSize   
        outputStarFile['rlnCoordinateZ'] = coords[:,2] / pixelSize

        # Angles
        outputStarFile['rlnAngleRot'] = coords[:,3]
        outputStarFile['rlnAngleTilt'] = coords[:,4]
        outputStarFile['rlnAnglePsi'] = coords[:,5]    
    else:
        outputStarFile['rlnCoordinateX'] = []
        outputStarFile['rlnCoordinateY'] = []  
        outputStarFile['rlnCoordinateZ'] = []

        # Angles
        outputStarFile['rlnAngleRot'] = []
        outputStarFile['rlnAngleTilt'] = []
        outputStarFile['rlnAnglePsi'] = [] 

    # Write 
    if tomoID == None:  savePath = os.path.join(outputDirectory, f'{specimen}.star')
    else:               savePath = os.path.join(outputDirectory, tomoID, f'{tomoID}_{specimen}.star')
    starfile.write( {'particles': pd.DataFrame(outputStarFile)}, savePath )

def write_copick_output(specimen, tomoID, finalPicks, outputDirectory='refinedCoPicks/ExperimentRuns', pickMethod='deepfinder', sessionID='0', knownTemplate=False):

    # Define the JSON structure
    json_data = {
        "pickable_object_name": specimen,
        "user_id": pickMethod,
        "session_id": sessionID,
        "run_name": tomoID,
        "voxel_spacing": None,
        "unit": "angstrom"
    }
    if not knownTemplate: json_data["trust_orientation"] = "false"
    
    json_data["points"] = []
    for ii in range(finalPicks.shape[0]):

        rotationMatrix = convert_euler_to_rotation_matrix(finalPicks[ii,3],  finalPicks[ii,4], finalPicks[ii,5])

        # Append to points data
        json_data['points'].append({
            "location": {"x": finalPicks[ii,0], "y": finalPicks[ii,1], "z": finalPicks[ii,2] },
            "transformation_": rotationMatrix,  # Convert matrix to list for JSON serialization
            "instance_id": 0,
            "score": 1.0
        })

    # Generate custom formatted JSON
    formatted_json = custom_format_json(json_data)

    # Save to file
    os.makedirs(os.path.join(outputDirectory, tomoID, 'Picks'), exist_ok=True)    
    savePath = os.path.join(outputDirectory, tomoID, 'Picks', f'{pickMethod}_{sessionID}_{specimen}.json')
    with open(savePath, 'w') as json_file:
        json_file.write(formatted_json)   

def custom_format_json(data):

    result = "{\n"
    for key, value in data.items():
        if key == "points":
            result += '   "{}": [\n'.format(key)
            for point in value:
                result += '      {\n'
                for p_key, p_value in point.items():
                    if p_key in ["location", "transformation_"]:
                        if p_key == "location":
                            loc_str = ', '.join(['"{}": {}'.format(k, v) for k, v in p_value.items()])
                            result += '         "{}": {{ {} }},\n'.format(p_key, loc_str)
                        if p_key == "transformation_":
                            trans_str = ',\n            '.join(['[{}]'.format(', '.join(map(str, row))) for row in p_value])
                            result += '         "{}": [\n            {}\n         ],\n'.format(p_key, trans_str)
                    else:
                        result += '         "{}": {},\n'.format(p_key, json.dumps(p_value))
                result = result.rstrip(',\n') + '\n      },\n'
            result = result.rstrip(',\n') + '\n   ]\n'
        else:
            result += '   "{}": {},\n'.format(key, json.dumps(value))
    result = result.rstrip(',\n') + '\n}'
    return result

def convert_euler_to_rotation_matrix(angleRot, angleTilt, anglePsi):

    rotation = R.from_euler('zyz', [angleRot, angleTilt, anglePsi], degrees=True)
    rotation_matrix = rotation.as_matrix()

    # Append a zero column to the right and zero row at the bottom
    new_column = np.zeros((3, 1))
    new_row = np.zeros((1, 4))
    rotation_matrix = np.vstack( (np.hstack((rotation_matrix, new_column)), new_row) )

    # Set the last element to 1
    rotation_matrix[-1, -1] = 1
    rotation_matrix = np.round(rotation_matrix,3)

    return rotation_matrix
