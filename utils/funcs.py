import numpy as np
import time
import yaml
import glob
import os
import cv2

from configs.default import values as default, main
from utils.yacs import Config


def save_config(values, metadata=None):
    metadata = values['main.config'] if metadata is None else metadata
    with open(metadata, "w", encoding='utf-8') as config:
        new_config = {}
        for k in values:
            if k in ['tg']:
                continue
            try:
                super_key, key, sub_key = k.split(".")
            except ValueError:
                sub_key = None
                try:
                    super_key, key = k.split(".")
                except ValueError:
                    super_key = k
                    key = None
            if key is not None:
                try:
                    _ = new_config[super_key]
                except KeyError:
                    new_config[super_key] = {}
                if sub_key is not None:
                    try:
                        _ = new_config[super_key][key]
                    except KeyError:
                        new_config[super_key][key] = {}
                    new_config[super_key][key][sub_key] \
                        = type_in_config(values[k])
                else:
                    new_config[super_key][key] = type_in_config(values[k])
            else:
                new_config[super_key] = type_in_config(values[k])
        yaml.dump(new_config, config, sort_keys=False, allow_unicode=True)


def load_config(window, archive):
    window['main.config'].update(archive)
    try:
        with open(archive, "r", encoding='utf-8') as config:
            archive = yaml.load(config, Loader=yaml.FullLoader)
    except FileNotFoundError:
        archive = default
    for super_key in archive:
        if super_key in ['tg']:
            continue
        for key in default[super_key]:
            if key in ['constant', 'scl', 'stop', 'ffmpeg']:
                continue
            elif isinstance(default[super_key][key], dict):
                for sub_key in default[super_key][key]:
                    window[f'{super_key}.{key}.{sub_key}'].update(
                        archive[super_key][key][sub_key])
            else:
                window[f'{super_key}.{key}'].update(archive[super_key][key])


def fastopenisp_mapping_config(values):
    result = {}
    result['module_enable_status'] = {}
    for k in values:
        if k.split('.')[0] == 'main':
            result[k] = values[k]
        if k.split('.')[0] == 'fast' and len(k.split('.')[1]) == 3:
            result['module_enable_status'][k.split('.')[1]] = values[k]
    result['hardware'] = {}
    result['hardware']['raw_width'] = values['main.width']
    result['hardware']['raw_height'] = values['main.height']
    result['hardware']['raw_bit_depth'] = values['main.bit_depth']
    for k in ['gbrg', 'bggr', 'rggb', 'grbg', 'srgb']:
        if values[f'main.{k}']:
            result['hardware']['bayer_pattern'] = k
            break
    result['dpc'] = {}
    result['dpc']['diff_threshold'] = values['fast.dpc_diff']
    result['blc'] = {}
    result['blc']['bl_r'] = values['fast.blc_bl']
    result['blc']['bl_gr'] = values['fast.blc_bl']
    result['blc']['bl_gb'] = values['fast.blc_bl']
    result['blc']['bl_b'] = values['fast.blc_bl']
    result['blc']['alpha'] = 0
    result['blc']['beta'] = 0
    result['lsc'] = {}
    result['lsc']['intensity'] = values['fast.lsc_intensity']
    result['aaf'] = None
    result['awb'] = {}
    result['awb']['gray_world'] = values['fast.awb_gw']
    result['awb']['r_gain'] = values['fast.awb_gain.r']
    result['awb']['gr_gain'] = values['fast.awb_gain.gr']
    result['awb']['gb_gain'] = values['fast.awb_gain.gb']
    result['awb']['b_gain'] = values['fast.awb_gain.b']
    result['cnf'] = {}
    result['cnf']['diff_threshold'] = values['fast.cnf_diff']
    result['cnf']['r_gain'] = values['fast.awb_gain.r']
    result['cnf']['b_gain'] = values['fast.awb_gain.b']
    result['cfa'] = {}
    for k in ['malvar', 'bilinear']:
        if values[f'fast.cfa_{k}']:
            result['cfa']['mode'] = k
            break
    result['ccm'] = {}
    result['ccm']['ccm'] = [[values['fast.ccm_matrix.r2r'],
                             values['fast.ccm_matrix.r2g'],
                             values['fast.ccm_matrix.r2b'],
                             values['fast.ccm_matrix.r2o']],
                            [values['fast.ccm_matrix.g2r'],
                             values['fast.ccm_matrix.g2g'],
                             values['fast.ccm_matrix.g2b'],
                             values['fast.ccm_matrix.g2o']],
                            [values['fast.ccm_matrix.b2r'],
                             values['fast.ccm_matrix.b2g'],
                             values['fast.ccm_matrix.b2b'],
                             values['fast.ccm_matrix.b2o']]]
    result['gac'] = {}
    result['gac']['gain'] = values['fast.gac_gain']
    result['gac']['gamma'] = values['fast.gac_gamma']
    result['csc'] = None
    result['nlm'] = {}
    result['nlm']['search_window_size'] = values['fast.nlm_search_window_size']
    result['nlm']['patch_size'] = values['fast.nlm_patch_size']
    result['nlm']['h'] = values['fast.nlm_h']
    result['bnf'] = {}
    result['bnf']['intensity_sigma'] = values['fast.bnf_intensity_sigma']
    result['bnf']['spatial_sigma'] = values['fast.bnf_spatial_sigma']
    result['ceh'] = {}
    result['ceh']['tiles'] = [values['fast.ceh_tile_width'],
                              values['fast.ceh_tile_height']]
    result['ceh']['clip_limit'] = float(values['fast.ceh_clip_limit'])
    result['eeh'] = {}
    result['eeh']['edge_gain'] = values['fast.eeh_gain']
    result['eeh']['flat_threshold'] = values['fast.eeh_flat_threshold']
    result['eeh']['edge_threshold'] = values['fast.eeh_edge_threshold']
    result['eeh']['delta_threshold'] = values['fast.eeh_delta_threshold']
    result['fcs'] = {}
    result['fcs']['delta_min'] = values['fast.fcs_delta_min']
    result['fcs']['delta_max'] = values['fast.fcs_delta_max']
    result['hsc'] = {}
    result['hsc']['hue_offset'] = values['fast.hsc_offset']
    result['hsc']['saturation_gain'] = values['fast.hsc_gain']
    result['bcc'] = {}
    result['bcc']['brightness_offset'] = values['fast.bcc_offset']
    result['bcc']['contrast_gain'] = values['fast.bcc_gain']
    result['scl'] = {}
    result['scl']['width'] = values['main.width']
    result['scl']['height'] = values['main.height']
    return Config(result)


def ycbcr_to_rgb(ycbcr_array):
    """ Convert YCbCr 3-channel array into sRGB array """
    assert ycbcr_array.dtype == np.uint8

    matrix = np.array([[298, 0, 409],
                       [298, -100, -208],
                       [298, 516, 0]], dtype=np.int32).T  # x256
    bias = np.array([-56992, 34784, -70688], dtype=np.int32).reshape(1, 1, 3)
    # x256

    ycbcr_array = ycbcr_array.astype(np.int32)
    rgb_array = np.right_shift(ycbcr_array @ matrix + bias, 8)
    rgb_array = np.clip(rgb_array, 0, 255)

    return rgb_array.astype(np.uint8)


def save_directory(save_dir, input_path):
    rgb_ext = [i[1] for i in main.constant.ext.rgb]
    rgb_path = os.path.join(save_dir,
                            os.path.basename(input_path)
                            .replace(input_path[-3:], rgb_ext[0][2:]))
    return rgb_path


def check_run_time(func):
    def new_func(*args, **kwargs):
        print("#" * 50)
        print(func.__name__)
        start_time = time.time()
        output = func(*args, **kwargs)
        end_time = time.time()
        print(f"total time : {end_time - start_time:.4f}")
        print("#" * 50)
        return output
    return new_func


def print_numpy(image, tag=''):
    if tag != '':
        print(f'{tag} =>', end='\t')
    print(f'Min.: {np.min(image):.2f}', end='\t')
    print(f'Max.: {np.max(image):.2f}', end='\t')
    print(f'Mean.: {np.mean(image):.2f}', end='\t')
    print(f'Std.: {np.sqrt(np.var(image)):.2f}', end='\t')
    print(f'Size: {image.shape}')


def type_in_config(value):
    if isinstance(value, bool):
        return value
    else:
        try:
            _dummy = float(value)
            if float.is_integer(_dummy):
                return int(value)
            else:
                return float(value)
        except ValueError:
            return str(value)


def video2frame(input_dir):
    vid_list = glob.glob(f"{input_dir}/*.mp4")
    if vid_list == glob.glob(f"{input_dir}/*"):
        print('Saving Frame First...')
        input_reader = cv2.VideoCapture(vid_list[0])
        i = 0
        while True:
            ret, frame = input_reader.read()
            if not ret:
                break
            cv2.imwrite(f'{input_dir}/frame_{i:03d}.png', frame)
            i += 1
        return True
    else:
        return False
