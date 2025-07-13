import PySimpleGUI
import os
import tqdm
import glob
import natsort

from easyisp.runners.runners import (EasyISPRunner, FastOpenISPRunner,
                                     NR3DRunner, CropRunner)
from configs.default import main, fast
from ui.layout import layout
from utils.funcs import (load_config, save_config, type_in_config, video2frame)


def read_config_at_ui(window):
    event, values = window.read()
    for k in values:
        values[k] = type_in_config(values[k])
    for k in fast.stop:
        window[f'fast.{k}'].update(False) if event == 'main.srgb' \
            else window[f'fast.{k}'].update(values[f'fast.{k}'])
    if event == 'main.config_load_button':
        load_config(window, values['main.config'])
    elif event == 'main.config_save_button':
        save_config(values)
    return event, values


def get_runner(event, values):
    if event in ['easy.run']:
        pp = EasyISPRunner(values)
    elif event in ['fast.run']:
        pp = FastOpenISPRunner(values)
    elif event in ['nr3d.run']:
        global input_vn, output_vn
        video2frame(values["main.in_path"])
        output_dir = values["main.out_path"]
        input_vn = f"{output_dir}/before3DNR.mp4"
        output_vn = f"{output_dir}/after3DNR.mp4"
        pp = NR3DRunner(values,
                        vn=[input_vn, output_vn])
    elif event in ['crop.run']:
        pp = CropRunner(values, window)
    else:
        raise NotImplementedError
    return pp


def get_input_list(values):
    ext_list = [i[1] for i in main.constant.ext.rgb] \
        if values['main.srgb'] else \
        [i[1] for i in main.constant.ext.raw]
    input_list = []
    for ext in ext_list:
        input_list += glob.glob(
            os.path.join(
                os.path.abspath(values['main.in_path']), ext))
    return natsort.natsorted(input_list)


def EasyISP(window):
    while True:
        event, values = read_config_at_ui(window)

        if '.run' in event:
            pp = get_runner(event, values)
            input_list = get_input_list(values)

            for input_npy in tqdm.tqdm(input_list):
                pp.Run(input_npy)

            if 'nr3d' in event:
                pp.input_writer.close()
                pp.output_writer.close()

            PySimpleGUI.popup_auto_close('완료되었습니다!',
                                         icon=main.constant.icon_path,
                                         auto_close_duration=5)

        if event == PySimpleGUI.WINDOW_CLOSED or '.quit' in event:
            break
    return window


if __name__ == '__main__':
    window = PySimpleGUI.Window(main.constant.name, layout=layout,
                                icon=main.constant.icon_path, location=(0, 0),
                                finalize=True)
    load_config(window, main.config)

    try:
        window = EasyISP(window)
    except Exception as e:
        print(e)

    _, values = window.read()
    save_config(values)
    window.close()
