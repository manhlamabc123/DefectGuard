from importlib.resources import files
import os, gdown

SRC_PATH = str(files('defectguard'))

IDS = {
    'deepjit': {
        'hyperparameters': '1US-qs1Ly9wfRADcEMLBtTa8Ao91wNwOv',
        'platform_within': '11Qjj84btTuqbYGpphmin0spMuGgJerNa',
        'platform_cross': '1BTo26TU2G58OsBxoM-EidyijfnQZXuc4',
        'platform_dictionary_within': '1C6nVSr0wLS8i8bH_IptCUKdqrdiSngcv',
        'platform_dictionary_cross': '1XY4J3bCKo7IWMXcA2DJqVzzAD8XOZi-b',
    },
    'cc2vec': {
        'qt_dictionary': '1GTgkEcZdwVDzp0Tq86Uch_j5f4assfSU',
        'dextended_qt_within': '1uuSeYee40Azw1jWD2ln287GZ49ApTMxL',
        'hyperparameters': '1Zim5j4eKfl84r4mGDmRELwAwg8oVQ5uJ',
        'cc2vec_qt_within': '1-ZQjygr6myPj4ml-VyyiyrGKiL0HV2Td',
    },
    'simcom': {
        'hyperparameters': '1Y9pt5EShp5Z2Q2Ff6EjHp0fxByXWViw6',
        'sim_qt_within': '1zee0mzb1bjnnim-WMer2K6e3WlWMNC0D',
        'sim_platform_within': '1SJ8UnaMQlaB58E7VsQWbHFmh2ms0QFg_',
        'sim_openstack_within': '1iJDpDLL19d_dp7mdjxu0ADqN25Hgyxuk',
        'sim_jdt_within': '1PPz385vq3cuuTf5pqM4k4c018rXoN1If',
        'sim_go_within': '1nknqQPbgJJXXCJ5pa4G27ymcEY2goxBq',
        'sim_gerrit_within': '1CmsiNXe5qXtEw6rG7IXLq2KVLslhOcij',
        'platform_dictionary': '19h6kUCiHXTsijXUEArxSx4afS4hdKrvx',
        'com_platform_within': '1KmUkYFVaH34kBA4pW8qXgv1JV9qCRtkx',
    },
    'lapredict': {
        'qt_within': '1HG-cscwvWAjWXlovqoyba1k5Do5NJh6b',
        'platform_within': '1kcWcD1PUDSksX7p_vKBlpVV20S3pcVQ8',
        'openstack_within': '1Y3bGUGoDEaQyUJAJ1x2-rdvLbcbdq-nz',
        'jdt_within': '1vjH9u7ObFPXuTtAdqNZAM47eeuDhl1B-',
        'go_within': '1r1mSvWvt4S93cZPI_j_bOKppBI-laLlI',
        'gerrit_within': '1484sBLghCpPd3XCpHt9Gqd_hvdq7TPP1',
    },
    'tlel': {
        'qt_within': '1ZvwEQ6lbb_43_JBgEB6VnRxR7HNQZlOk',
        'platform_within': '1vS26ng_kZ5gdYESyWrfciMacXz74AzhZ',
        'openstack_within': '1yCOI_5inFnxH1EDS2JpA282UN7Zc1AXV',
        'jdt_within': '1GUEC7kFCybuoEetr-1Tis_6EmaWXgWwG',
        'go_within': '1siGmkBSq5qcuoxnhxo2Gc2_IhrVnLmWh',
        'gerrit_within': '1CI326L7vwokRXxwRdufzOvKtciPUj_TX',
    },
    'jitline': '',
}

def create_download_list(model_name, dataset, project):
    download_list = []
    dictionary = f'{dataset}_dictionary_{project}'
    version = f'{dataset}_{project}'

    if model_name == 'simcom':
        sim_dataset = f'sim_{version}'
        com_dataset = f'com_{version}'
        download_list.append(sim_dataset)
        download_list.append(com_dataset)
        download_list.append('hyperparameters')
        download_list.append(dictionary)
    elif model_name == 'cc2vec':
        cc2vec_dataset = f'cc2vec_{version}'
        dextended_dataset = f'dextended_{version}'
        download_list.append(cc2vec_dataset)
        download_list.append(dextended_dataset)
        download_list.append('hyperparameters')
        download_list.append(dictionary)
    elif model_name == 'deepjit':
        download_list.append(version)
        download_list.append('hyperparameters')
        download_list.append(dictionary)
    else:
        download_list.append(version)
    
    return download_list

def download_file(file_id, folder_path):
    if not os.path.isfile(folder_path):
        gdown.download(
            f'https://drive.google.com/uc?/export=download&id={file_id}',
            output=folder_path
            )

def download_folder(model_name, dataset, project):
    # Check if the file exists locally
    folder_path = f'{SRC_PATH}/models/{model_name}'
    print(f"Folder's path: {folder_path}")

    if not os.path.exists(folder_path):
        # File doesn't exist, download it
        print(f"'{model_name}' does not exist locally. Downloading...")

        # Create the directory if it doesn't exist
        print(f"Directory: {folder_path}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Download model's metadata
    download_list = create_download_list(model_name, dataset, project)
    print(f"Download list: {download_list}")
    for item in download_list:
        download_file(IDS[model_name][item], f'{folder_path}/{item}')

    print(f"'{model_name}' downloaded.")