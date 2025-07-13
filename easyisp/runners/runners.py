import os
import cv2
import numpy as np
import sys
import imageio
import copy
import math
import importlib

from typing import Dict, Optional, Union
from collections import OrderedDict
from multiprocessing import Process
from mmengine.config import Config, ConfigDict

from easyisp.registry import RUNNERS
from easyisp.transforms import Compose
from easyisp.runners.modules.basic_module import MODULE_DEPENDENCIES

from utils.funcs import (save_config, save_directory,
                         fastopenisp_mapping_config, ycbcr_to_rgb)
from utils.yacs import Config as FastOpenISPConfig

ConfigType = Union[Dict, Config, ConfigDict, FastOpenISPConfig]


@RUNNERS.register_module()
class EasyISPRunner:
    def __init__(self,
                 config: Optional[ConfigType] = None):
        self.cfg = config
        self.bayer = {
            'gbrg': self.cfg['main.gbrg'],
            'bggr': self.cfg['main.bggr'],
            'rggb': self.cfg['main.rggb'],
            'grbg': self.cfg['main.grbg'],
            'srgb': self.cfg['main.srgb'],
        }
        self.bayer = list(self.bayer.keys())[
            list(self.bayer.values()).index(True)]
        self.ccm = [[self.cfg['easy.ccm_matrix.r2r'],
                     self.cfg['easy.ccm_matrix.r2g'],
                     self.cfg['easy.ccm_matrix.r2b']],
                    [self.cfg['easy.ccm_matrix.g2r'],
                     self.cfg['easy.ccm_matrix.g2g'],
                     self.cfg['easy.ccm_matrix.g2b']],
                    [self.cfg['easy.ccm_matrix.b2r'],
                     self.cfg['easy.ccm_matrix.b2g'],
                     self.cfg['easy.ccm_matrix.b2b']]]
        self.device = {
            'cpu': self.cfg['main.cpu'],
            'cuda': self.cfg['main.cuda'],
        }
        self.device = list(self.device.keys())[
            list(self.device.values()).index(True)]
        self.pipeline = self._set_pipeline([])

    def _set_pipeline(self, p_l):
        p_l.append(dict(type='LoadNumpyArray',
                        device=self.device,
                        size=(self.cfg['main.height'],
                              self.cfg['main.width'])))
        if self.cfg['easy.blc']:
            p_l.append(dict(type='BlackWhiteLevel',
                            black_level=int(self.cfg['easy.blc_bl']),
                            white_level=2 ** (int(
                                self.cfg['main.bit_depth'])) - 1,
                            device=self.device))
        if self.cfg['easy.cfa']:
            p_l.append(dict(type='Bayer2RGB', bayer=self.bayer,
                            device=self.device))
        p_l.append(dict(type='AstypeNumpy', input_type='uint16',
                        output_type='float', device=self.device))
        if self.cfg['easy.awb']:
            p_l.append(dict(type='AutoWhiteBalance',
                            device=self.device,
                            gw=self.cfg['easy.awb_gw'],
                            gain={
                                "r": self.cfg['easy.awb_gain.r'],
                                "g": self.cfg['easy.awb_gain.g'],
                                "b": self.cfg['easy.awb_gain.b']
                                 },))
        if self.cfg['easy.ccm']:
            p_l.append(dict(type='ColorCorrectionMatrix', ccm=self.ccm,
                            device=self.device))
        if self.cfg['easy.gac']:
            p_l.append(dict(type='GammaCorrection',
                            digital_gain=self.cfg['easy.gac_gain'],
                            gamma=self.cfg['easy.gac_gamma'],
                            device=self.device))
        p_l.append(dict(type='AstypeNumpy', input_type='float',
                        output_type='uint8', device=self.device))
        p_l.append(dict(type='CvtColor', input_type='rgb',
                        output_type='bgr', device=self.device))
        if self.device == 'cuda':
            p_l.append(dict(type='Torch2Cv2Image'))
        return p_l

    def Run(self, raw_path):
        self.pipeline = Compose(self.pipeline)
        results = dict(img_path=raw_path)
        results = self.pipeline(results)
        bgr_image = results['img_frame']
        rgb_path = save_directory(self.cfg['main.out_path'], raw_path)

        if rgb_path.isalpha():
            cv2.imwrite(rgb_path, bgr_image)
        else:
            extension = os.path.splitext(rgb_path)[1]
            result, encoded_img = cv2.imencode(extension, bgr_image)
            if result:
                with open(rgb_path, mode='w+b') as f:
                    encoded_img.tofile(f)

        save_config(self.cfg, f'{self.cfg["main.out_path"]}/metadata.yaml')


@RUNNERS.register_module()
class FastOpenISPRunner:
    """ Core fast-openISP pipeline """
    def __init__(self, config):
        """
        :param cfg: yacs.Config object, configurations about camera specs and
        module parameters
        """
        self.save_cfg = config
        self.cfg = fastopenisp_mapping_config(config)

        saturation_values = self.get_saturation_values()
        with self.cfg.unfreeze():
            self.cfg.saturation_values = saturation_values

        self.modules = self.get_modules()
        self.device = 'cpu'
        self.pipeline = self._set_pipeline()

    def _set_pipeline(self):
        if not self.cfg['main.srgb']:
            p_l = [
                dict(type='LoadNumpyArray',
                     device=self.device,
                     size=(self.cfg['main.height'], self.cfg['main.width'])),
                dict(type='ExecuteFastOpenISP', isp=self),
                dict(type='CvtColor', input_type='rgb',
                     output_type='bgr', device=self.device)
            ]
        else:
            p_l = [
                dict(type='LoadCv2Image'),
                dict(type='ExecuteFastOpenISP', isp=self),
                dict(type='CvtColor', input_type='rgb',
                     output_type='bgr', device=self.device)
            ]
        return p_l

    def Run(self, raw_path):
        self.pipeline = Compose(self.pipeline)
        results = dict(img_path=raw_path)
        results = self.pipeline(results)
        bgr_image = results['img_frame']
        rgb_path = save_directory(self.cfg['main.out_path'], raw_path)

        if rgb_path.isalpha():
            cv2.imwrite(rgb_path, bgr_image)
        else:
            extension = os.path.splitext(rgb_path)[1]
            result, encoded_img = cv2.imencode(extension, bgr_image)
            if result:
                with open(rgb_path, mode='w+b') as f:
                    encoded_img.tofile(f)

        save_config(self.save_cfg,
                    f'{self.cfg["main.out_path"]}/metadata.yaml')

    def get_saturation_values(self):
        """
        Get saturation pixel values in different stages in the pipeline.
        Raw stage: dataflow before the BLC modules (not included)
        HDR stage: dataflow after the BLC modules (included) and before the
        bit-depth compression
            module, i.e., Gamma in openISP (not included)
        SDR stage: dataflow after the Gamma module (included)
        """
        raw_max_value = 2 ** self.cfg.hardware.raw_bit_depth - 1
        sdr_max_value = 255

        # Saturation values should be carefully calculated if BLC module is
        # activated
        if 'blc' in self.cfg.module_enable_status:
            blc = self.cfg.blc
            hdr_max_r = raw_max_value - blc.bl_r
            hdr_max_b = raw_max_value - blc.bl_b
            hdr_max_gr = int(
                raw_max_value - blc.bl_gr + hdr_max_r * blc.alpha / 1024)
            hdr_max_gb = int(
                raw_max_value - blc.bl_gb + hdr_max_b * blc.beta / 1024)
            hdr_max_value = max(hdr_max_r, hdr_max_b, hdr_max_gr, hdr_max_gb)
        else:
            hdr_max_value = raw_max_value

        return Config({'raw': raw_max_value,
                       'hdr': hdr_max_value,
                       'sdr': sdr_max_value})

    def get_modules(self):
        """ Get activated ISP modules according to the configuration """
        if os.path.dirname(__file__) not in sys.path:
            sys.path.insert(0, os.path.dirname(__file__))

        enabled_modules = tuple(m for m, en in
                                self.cfg.module_enable_status.items() if en)

        modules = OrderedDict()
        for module_name in enabled_modules:
            package = importlib.import_module('modules.{}'.format(module_name))
            module_cls = getattr(package, module_name.upper())
            module = module_cls(self.cfg)

            for m in MODULE_DEPENDENCIES.get(module_cls.__name__, []):
                if m not in enabled_modules:
                    raise RuntimeError(
                        '{} is unavailable when {} is deactivated'.format(
                            module_name, m)
                    )

            modules[module_name] = module

        return modules

    def execute(self, bayer, save_intermediates=False, verbose=True):
        """
        ISP pipeline execution
        :param bayer: input Bayer array, np.ndarray(H, W)
        :param save_intermediates: whether to save intermediate results from
        all ISP modules
        :param verbose: whether to print timing messages
        :return:
            data: a dict containing results from different domains
            (Bayer, RGB, and YCbCr)
                and the final RGB output (data['output'])
            intermediates: a dict containing intermediate results
            if save_intermediates=True,
                otherwise a empty dict
        """

        def print_(*args, **kwargs):
            return print(*args, **kwargs) if verbose else None

        # pipeline_start = time.time()

        data = OrderedDict(bayer=bayer, rgb_image=bayer)
        intermediates = OrderedDict()

        for module_name, module in self.modules.items():
            # start = time.time()
            # print_(
            #     'Executing {}... '.format(module_name), end='', flush=True)

            module.execute(data)
            if save_intermediates:
                intermediates[module_name] = copy.copy(data)

            # print_('Done. Elapsed {:.3f}s'.format(time.time() - start))

        data['output'] = self.get_output(data)
        # print_(
        #     'Pipeline elapsed {:.3f}s'.format(time.time() - pipeline_start))

        return data, intermediates

    def get_output(self, data):
        """
        Post-process the pipeline result to get the final output
        :param data: argument returned by self.execute()
        :return: displayable result: np.ndarray(H, W, 3) in np.uint8 dtype
        """
        if 'y_image' in data and 'cbcr_image' in data:
            ycbcr_image = np.dstack([data['y_image'][..., None],
                                     data['cbcr_image']])
            output = ycbcr_to_rgb(ycbcr_image)
        elif 'rgb_image' in data:
            output = data['rgb_image']
            if output.dtype != np.uint8:
                output = output.astype(np.float32)
                output = (255 * output / self.cfg.saturation_values.hdr)\
                    .astype(np.uint8)
        elif 'bayer' in data:
            # actually not an RGB image, looks very dark for most cameras
            output = data['bayer']
            output = output.astype(np.float32)
            output = (255 * output / self.cfg.saturation_values.raw)\
                .astype(np.uint8)
        else:
            raise NotImplementedError

        return output

    def run(self, raw_path, save_dir, load_raw_fn, suffix=''):
        """
        A higher level API that writes ISP result into disk
        :param raw_path: path to the raw file to be processed
        :param save_dir: directory to save the output
        (shares the same filename as the input)
        :param load_raw_fn: function to load the Bayer array from the raw_path
        :param suffix: suffix to added to the output filename
        """
        import cv2

        bayer = load_raw_fn(raw_path)
        data, _ = self.execute(bayer, save_intermediates=False, verbose=False)
        output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)

        filename = os.path.splitext(os.path.basename(raw_path))[0]
        save_path = os.path.join(save_dir, '{}.png'.format(filename + suffix))
        cv2.imwrite(save_path, output)

    def batch_run(self, raw_paths, save_dirs, load_raw_fn, suffixes='',
                  num_processes=1):
        """
        Batch version of self.run via multiprocessing
        :param raw_paths: list of paths to the raw files to be executed
        :param save_dirs: list of directories to save the outputs.
        If given a string, it will be copied to a N-element list,
        where N is the number of paths in raw_paths
        :param load_raw_fn: function to load the Bayer array from the raw_path
        :param suffixes: a list of suffixes to added to the output filenames
        :param num_processes: number of processes in multiprocessing
        """
        num_files = len(raw_paths)
        num_batches = math.ceil(num_files / num_processes)

        if not isinstance(save_dirs, (list, tuple)):
            save_dirs = [save_dirs for _ in range(num_files)]
        if not isinstance(suffixes, (list, tuple)):
            suffixes = [suffixes for _ in range(num_files)]

        for batch_id in range(num_batches):
            indices = [batch_id * num_processes + rank
                       for rank in range(num_processes)]
            indices = [i for i in indices if i < num_files]
            batch_size = len(indices)

            raw_paths_batch = [raw_paths[i] for i in indices]
            save_dirs_batch = [save_dirs[i] for i in indices]
            suffixes_batch = [suffixes[i] for i in indices]

            pool = []
            for rank in range(batch_size):
                pool.append(
                    Process(target=self.run,
                            kwargs={'raw_path': raw_paths_batch[rank],
                                    'save_dir': save_dirs_batch[rank],
                                    'load_raw_fn': load_raw_fn,
                                    'suffix': suffixes_batch[rank]})
                )

            for p in pool:
                p.start()

            for p in pool:
                p.join()


class NR3DRunner:
    def __init__(self, config, vn):
        super().__init__()
        self.cfg = config
        self.stack_frames = None
        self.diff_frames = None

        self.device = {
            'cpu': self.cfg['main.cpu'],
            'cuda': self.cfg['main.cuda'],
        }
        self.device = list(self.device.keys())[
            list(self.device.values()).index(True)]

        try:
            os.mkdir(os.path.abspath(self.cfg['main.out_path']))
        except FileExistsError:
            pass

        self.input_writer = imageio.get_writer(
            vn[0], fps=float(self.cfg['nr3d.fps']), codec='libx264',
            ffmpeg_params=['-qp', '0'], macro_block_size=1)
        self.output_writer = imageio.get_writer(
            vn[1], fps=float(self.cfg['nr3d.fps']), codec='libx264',
            ffmpeg_params=['-qp', '0'], macro_block_size=1)

        self.pipeline = self._set_pipeline()

    def _set_pipeline(self, p_l=[]):
        p_l.append(dict(type='LoadCv2Frame',
                        input_writer=self.input_writer))
        if self.cfg['nr3d.filter_nr3d']:
            p_l.append(dict(type='FrameFilter', config=self.cfg))
        if not self.cfg['nr3d.filter']:
            p_l.append(dict(type='StackFrames',
                            stack_frames=self.stack_frames,
                            num_frames=self.cfg['nr3d.num_frames'],
                            device=self.device))
            p_l.append(dict(type='DeAfterImage',
                            diff_frames=self.diff_frames,
                            threshold=self.cfg['nr3d.threshold'],
                            device=self.device))
        if self.cfg['nr3d.nr3d_filter']:
            p_l.append(dict(type='FrameFilter', config=self.cfg))
        return p_l

    def Run(self, rgb_path):
        self.pipeline = Compose(self.pipeline)
        results = dict(img_path=rgb_path)
        results = self.pipeline(results)
        rgb_image = results['img_frame']
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        rgb_path = save_directory(self.cfg['main.out_path'], rgb_path)

        if rgb_path.isalpha():
            cv2.imwrite(rgb_path, bgr_image)
        else:
            extension = os.path.splitext(rgb_path)[1]
            result, encoded_img = cv2.imencode(extension, bgr_image)
            if result:
                with open(rgb_path, mode='w+b') as f:
                    encoded_img.tofile(f)

        self.output_writer.append_data(rgb_image)
        self.stack_frames = results['stack_frames']
        self.diff_frames = results['diff_frames']

        save_config(self.cfg, f'{self.cfg["main.out_path"]}/metadata.yaml')

    def imwrite(self, rgb_path, bgr_image):
        rgb_path = save_directory(self.cfg['main.out_path'], rgb_path)

        if rgb_path.isalpha():
            cv2.imwrite(rgb_path, bgr_image)
        else:
            extension = os.path.splitext(rgb_path)[1]
            result, encoded_img = cv2.imencode(extension, bgr_image)
            if result:
                with open(rgb_path, mode='w+b') as f:
                    encoded_img.tofile(f)


class CropRunner:
    def __init__(self, config, window):
        super().__init__()
        self.cfg = config
        self.window = window

        self.device = {
            'cpu': self.cfg['main.cpu'],
            'cuda': self.cfg['main.cuda'],
        }
        self.device = list(self.device.keys())[
            list(self.device.values()).index(True)]

        self.roi = None

    def _set_pipeline(self):
        p_l = [
            dict(type='LoadCv2Image'),
            dict(type='CvtColor', input_type='rgb',
                 output_type='bgr', device=self.device),
            dict(type='CropCv2Image', roi=self.roi, window=self.window),
        ]
        return p_l

    def Run(self, rgb_path):
        pipeline = Compose(self._set_pipeline())
        results = dict(img_path=rgb_path)
        results = pipeline(results)
        bgr_image = results['img_frame']

        rgb_path = save_directory(self.cfg['main.out_path'], rgb_path)

        if rgb_path.isalpha():
            cv2.imwrite(rgb_path, bgr_image)
        else:
            extension = os.path.splitext(rgb_path)[1]
            result, encoded_img = cv2.imencode(extension, bgr_image)
            if result:
                with open(rgb_path, mode='w+b') as f:
                    encoded_img.tofile(f)
        print("after:", results['img_roi'])
        self.roi = results['img_roi']
