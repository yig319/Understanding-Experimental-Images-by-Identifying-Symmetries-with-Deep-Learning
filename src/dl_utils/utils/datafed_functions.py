import os
import json # For dealing with metadata
from datafed.CommandLib import API

def datafed_create_collection(collection_name, parent_id=None):
    df_api = API()
    coll_resp = df_api.collectionCreate(collection_name, parent_id=parent_id)
    return coll_resp

def visualize_datafed_collection(collection_id, max_count=100):
    df_api = API()
    item_list = []
    for item in list(df_api.collectionItemsList(collection_id, count=max_count)[0].item):
        print(item)
        item_list.append(item)
    return item_list

def datafed_upload(file_path, parent_id, metadata=None, wait=True):
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