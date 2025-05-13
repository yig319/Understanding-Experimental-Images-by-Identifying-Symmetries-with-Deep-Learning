import os
import json # For dealing with metadata
from datafed.CommandLib import API

def datafed_create_collection(collection_name, parent_id=None):
    """
    Creates a new collection in DataFed.

    Parameters:
        collection_name (str): Name of the new collection.
        parent_id (str, optional): ID of the parent collection where the new collection will be placed.
                                   If None, the collection is created at the root level.

    Returns:
        Collection creation response object from DataFed API containing the new collection's details.
    """
    df_api = API()
    coll_resp = df_api.collectionCreate(collection_name, parent_id=parent_id)
    return coll_resp


def visualize_collection(collection_id, max_count=100):
    """
    Prints and returns a list of items in a specified DataFed collection.

    Parameters:
    - collection_id (str): ID of the collection to visualize.

    Returns:
    - list: A list of items within the specified collection.
    """

    df_api = API()
    item_list = []
    for item in list(df_api.collectionItemsList(collection_id, count=max_count)[0].item):
        print(item)
        item_list.append(item)
    return item_list

def datafed_upload(file_path, parent_id, metadata=None, wait=True):
    """
    Uploads a single file to a DataFed collection with optional metadata.

    Parameters:
    - file_path (str): Path to the file to upload.
    - parent_id (str): ID of the DataFed collection to upload the file into.
    - metadata (dict, optional): Dictionary of metadata to attach to the file. Defaults to None.
    - wait (bool, optional): If True, waits for the Globus transfer to complete. Defaults to True.

    Returns:
    - Response from DataFed API after file upload request.
    """
    df_api = API()

    file_name = os.path.basename(file_path)
    dc_resp = df_api.dataCreate(file_name, metadata=json.dumps(metadata), parent_id=parent_id)
    rec_id = dc_resp[0].data[0].id
    put_resp = df_api.dataPut(rec_id, file_path, wait=wait)
    print(put_resp)
    
def datafed_download(file_path, file_id, wait=True):
    df_api = API()
    get_resp = df_api.dataGet([file_id], # currently only accepts a list of IDs / aliases
                              file_path, # directory where data should be downloaded
                              orig_fname=True, # do not name file by its original name
                              wait=wait, # Wait until Globus transfer completes
    )
    print(get_resp)

def datafed_update_record(record_id, metadata):
    df_api = API()
    du_resp = df_api.dataUpdate(record_id,
                                metadata=json.dumps(metadata),
                                metadata_set=True,
                                )
    print(du_resp)
    

def datafed_upload_folder(folder_path, parent_id=None, metadata=None, wait=True, rule=None):
    """
    Recursively uploads folders and selectively uploads files based on a provided rule.

    Steps:
    1. Creates a DataFed collection for the given folder.
    2. Applies a selection rule (if provided) to filter files before upload.
    3. Uploads selected files into the collection.
    4. Recursively handles subfolders by creating sub-collections and repeating the process.

    Parameters:
    - folder_path (str): Path to the root folder to recursively upload.
    - parent_id (str, optional): ID of the parent DataFed collection. Defaults to None.
    - metadata (dict, optional): Metadata dictionary attached to each file uploaded. Defaults to None.
    - wait (bool, optional): If True, waits for uploads to complete before proceeding. Defaults to True.
    - rule (callable, optional): Function to select which files to upload. Should accept `(files, folder_path)`
      as arguments and return a filtered list of filenames. Defaults to None (uploads all files).

    Returns:
    - str: Collection ID of the newly created DataFed collection.
    """
    
    folder_name = os.path.basename(folder_path.rstrip('/'))

    # Create collection for the current folder
    coll_resp = datafed_create_collection(folder_name, parent_id=parent_id)
    collection_id = coll_resp[0].coll[0].id
    print(f"Created collection '{folder_name}' with ID: {collection_id}")

    entries = os.listdir(folder_path)

    # Separate files and directories
    files = [e for e in entries if os.path.isfile(os.path.join(folder_path, e))]
    dirs = [e for e in entries if os.path.isdir(os.path.join(folder_path, e))]

    # Apply rule (if provided) on files only
    if rule is not None:
        files = rule(files, folder_path)

    # Upload selected files
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            print(f"Uploading file '{file}' to collection '{folder_name}'...")
            datafed_upload(file_path, parent_id=collection_id, metadata=metadata, wait=wait)
            print(f"Uploaded '{file}'")
        except Exception as e:
            print(f"Error uploading file '{file}': {e}")

    # Recursively handle sub-directories
    for dir in dirs:
        dir_path = os.path.join(folder_path, dir)
        datafed_upload_folder(dir_path, parent_id=collection_id, metadata=metadata, wait=wait, rule=rule)

    print(f"Finished uploading '{folder_name}'.")
    return collection_id
