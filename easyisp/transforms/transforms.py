from typing import Dict, Tuple

import os
import sys
import cv2
import numpy as np
import torch

from ..registry import TRANSFORMS
from .base import BaseTransform


@TRANSFORMS.register_module()
class LoadNumpyArray(BaseTransform):
    def __init__(self, device: str, size: Tuple[int, int]) -> None:
        self.device = device
        self.size = size

    def transform(self, results: dict) -> dict:
        img_path = results['img_path']
        if os.path.splitext(img_path)[1] == '.npy':
            img_frame = np.load(img_path)
        elif os.path.splitext(img_path)[1].lower() == '.raw':
            img_frame = np.fromfile(img_path, dtype='uint16', sep='')
            img_frame = img_frame.reshape(self.size)
        if self.device == 'cpu':
            pass
        elif self.device == 'cuda':
            img_frame = torch.from_numpy(img_frame).to(
                self.device).unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError
        results['img_frame'] = img_frame
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class BlackWhiteLevel(BaseTransform):
    def __init__(self,
                 black_level: int = 240,
                 white_level: int = 4095,
                 device: str = 'cuda') -> None:
        self.black_level = black_level
        self.white_level = white_level
        self.device = device

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']  # W H, np.uint16
        if self.device == 'cpu':
            img_frame = img_frame.astype(np.float32)

            # black white level
            img_frame = (img_frame - self.black_level) / \
                (self.white_level - self.black_level)  # black level
            img_frame = np.clip(img_frame, 0, 1)  # clip

            # float64 to uint16
            img_frame = img_frame * 65535
            img_frame = img_frame.astype(np.uint16)
        elif self.device == 'cuda':
            img_frame = img_frame.to(torch.float32)

            img_frame = img_frame.sub_(
                self.black_level).div_(
                self.white_level - self.black_level).clamp(0, 1)
        else:
            raise NotImplementedError
        results['img_frame'] = img_frame
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(black_level={self.black_level}, '
        repr_str += f'white_level={self.white_level})'
        return repr_str


@TRANSFORMS.register_module()
class Bayer2RGB(BaseTransform):
    def __init__(self,
                 bayer: str = 'gbrg',
                 device: str = 'cuda') -> None:
        self.bayer = bayer
        self.device = device
        if self.device == 'cuda':
            self.g_at_r = (
                torch.Tensor(
                    [
                        [0, 0, -1, 0, 0],
                        [0, 0, 2, 0, 0],
                        [-1, 2, 4, 2, -1],
                        [0, 0, 2, 0, 0],
                        [0, 0, -1, 0, 0],
                    ]
                ).to(self.device)
                .float()
                .view(1, 1, 5, 5)
                / 8
            ).to(self.device)

            self.g_at_b = self.g_at_r.clone()

            self.r_at_g1 = (
                torch.Tensor(
                    [
                        [0, 0, 0.5, 0, 0],
                        [0, -1, 0, -1, 0],
                        [-1, 4, 5, 4, -1],
                        [0, -1, 0, -1, 0],
                        [0, 0, 0.5, 0, 0],
                    ],
                ).to(self.device)
                .float()
                .view(1, 1, 5, 5)
                / 8
            )

            self.r_at_g2 = (
                torch.Tensor(
                    [
                        [0, 0, -1, 0, 0],
                        [0, -1, 4, -1, 0],
                        [0.5, 0, 5, 0, 0.5],
                        [0, -1, 4, -1, 0],
                        [0, 0, -1, 0, 0],
                    ],
                ).to(self.device)
                .float()
                .view(1, 1, 5, 5)
                / 8
            )

            self.r_at_b = (
                torch.Tensor(
                    [
                        [0, 0, -1.5, 0, 0],
                        [0, 2, 0, 2, 0],
                        [-1.5, 0, 6, 0, -1.5],
                        [0, 2, 0, 2, 0],
                        [0, 0, -1.5, 0, 0],
                    ],
                ).to(self.device)
                .float()
                .view(1, 1, 5, 5)
                / 8
            )

            self.b_at_g1 = self.r_at_g2.clone()
            self.b_at_g2 = self.r_at_g1.clone()
            self.b_at_r = self.r_at_b.clone()

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']  # W H, uint16
        if self.device == 'cpu':
            if self.bayer == 'gbrg':
                img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerGR2RGB)
            elif self.bayer == 'bggr':
                img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerRG2RGB)
            elif self.bayer == 'rggb':
                img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerBG2RGB)
            elif self.bayer == 'grbg':
                img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerGB2RGB)
            else:
                raise NotImplementedError
        elif self.device == 'cuda':
            img_frame = self.malvar(img_frame)
        else:
            raise NotImplementedError
        results['img_frame'] = img_frame
        # print(f'{self.__repr__()} Completed!')
        return results

    def pack_raw(self, x):
        if self.bayer == 'gbrg':
            g10 = x[:, :, 0::2, 0::2]
            b11 = x[:, :, 0::2, 1::2]
            r00 = x[:, :, 1::2, 0::2]
            g01 = x[:, :, 1::2, 1::2]
        elif self.bayer == 'bggr':
            b11 = x[:, :, 0::2, 0::2]
            g10 = x[:, :, 0::2, 1::2]
            g01 = x[:, :, 1::2, 0::2]
            r00 = x[:, :, 1::2, 1::2]
        elif self.bayer == 'rggb':
            r00 = x[:, :, 0::2, 0::2]
            g01 = x[:, :, 0::2, 1::2]
            g10 = x[:, :, 1::2, 0::2]
            b11 = x[:, :, 1::2, 1::2]
        elif self.bayer == 'grbg':
            g01 = x[:, :, 0::2, 0::2]
            r00 = x[:, :, 0::2, 1::2]
            b11 = x[:, :, 1::2, 0::2]
            g10 = x[:, :, 1::2, 1::2]
        else:
            raise NotImplementedError
        return torch.cat([r00, g01, g10, b11], dim=1)

    def malvar(self, x):
        x = self.pack_raw(x)

        self.g_at_r = self.g_at_r.to(x.device)
        self.g_at_b = self.g_at_b.to(x.device)
        self.r_at_g1 = self.r_at_g1.to(x.device)
        self.r_at_g2 = self.r_at_g2.to(x.device)
        self.r_at_b = self.r_at_b.to(x.device)
        self.b_at_g1 = self.b_at_g1.to(x.device)
        self.b_at_g2 = self.b_at_g2.to(x.device)
        self.b_at_r = self.b_at_r.to(x.device)

        x = torch.nn.functional.pixel_shuffle(x, 2)
        g00 = torch.nn.functional.conv2d(x, self.g_at_r, padding=2, stride=2)
        g01 = x[:, :, 0::2, 1::2]
        g10 = x[:, :, 1::2, 0::2]
        g11 = torch.nn.functional.conv2d(x.flip(dims=(2, 3)),
                                         self.g_at_b, padding=2, stride=2)
        g11 = g11.flip(dims=(2, 3))
        g = torch.nn.functional.pixel_shuffle(
            torch.cat([g00, g01, g10, g11], dim=1), 2)

        r00 = x[:, :, 0::2, 0::2]
        r01 = torch.nn.functional.conv2d(x.flip(dims=(3,)),
                                         self.r_at_g1, padding=2, stride=2)
        r01 = r01.flip(dims=(3,))
        r10 = torch.nn.functional.conv2d(x.flip(dims=(2,)),
                                         self.r_at_g2, padding=2, stride=2)
        r10 = r10.flip(dims=(2,))
        r11 = torch.nn.functional.conv2d(x.flip(dims=(2, 3)),
                                         self.r_at_b, padding=2, stride=2)
        r11 = r11.flip(dims=(2, 3))
        r = torch.nn.functional.pixel_shuffle(
            torch.cat([r00, r01, r10, r11], dim=1), 2)

        b00 = torch.nn.functional.conv2d(x, self.b_at_r, padding=2, stride=2)
        b01 = torch.nn.functional.conv2d(x.flip(dims=(3,)),
                                         self.b_at_g1, padding=2, stride=2)
        b01 = b01.flip(dims=(3,))
        b10 = torch.nn.functional.conv2d(x.flip(dims=(2,)),
                                         self.b_at_g2, padding=2, stride=2)
        b10 = b10.flip(dims=(2,))
        b11 = x[:, :, 1::2, 1::2]
        b = torch.nn.functional.pixel_shuffle(
            torch.cat([b00, b01, b10, b11], dim=1), 2)
        return torch.cat([r, g, b], dim=1)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(bayer={self.bayer})'
        return repr_str


@TRANSFORMS.register_module()
class AstypeNumpy(BaseTransform):
    def __init__(self,
                 input_type: str,
                 output_type: str,
                 device: str) -> None:
        self.input_type = input_type
        self.output_type = output_type
        self.device = device

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']
        if self.device == 'cpu':
            if self.input_type == 'uint16' and self.output_type == 'float':
                img_frame = img_frame.astype(np.float64)
                img_frame /= 65535
            elif self.input_type == 'float' and self.output_type == 'uint8':
                img_frame *= 255
                img_frame = img_frame.astype(np.uint8)
            else:
                raise NotImplementedError
        elif self.device == 'cuda':
            self.input_type = 'float'
            self.output_type = 'float'
        else:
            raise NotImplementedError
        results['img_frame'] = img_frame
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(input_type={self.input_type}, '
        repr_str += f'output_type={self.output_type})'
        return repr_str


@TRANSFORMS.register_module()
class CvtColor(BaseTransform):
    def __init__(self,
                 input_type: str,
                 output_type: str,
                 device: str) -> None:
        self.input_type = input_type
        self.output_type = output_type
        self.device = device

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']
        if self.device == 'cpu':
            if self.input_type == 'bgr' and self.output_type == 'rgb':
                img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
            elif self.input_type == 'rgb' and self.output_type == 'bgr':
                img_frame = cv2.cvtColor(img_frame, cv2.COLOR_RGB2BGR)
            else:
                raise NotImplementedError
        elif self.device == 'cuda':
            if self.input_type == 'bgr' and self.output_type == 'rgb':
                pass
            elif self.input_type == 'rgb' and self.output_type == 'bgr':
                pass
            else:
                raise NotImplementedError
            img_frame = torch.cat([img_frame[:, 2, :, :],
                                   img_frame[:, 1, :, :],
                                   img_frame[:, 0, :, :]])
        else:
            raise NotImplementedError
        results['img_frame'] = img_frame
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(input_type={self.input_type}, '
        repr_str += f'output_type={self.output_type})'
        return repr_str


@TRANSFORMS.register_module()
class AutoWhiteBalance(BaseTransform):
    def __init__(self,
                 device: str,
                 gw: bool,
                 gain: Dict[str, float]) -> None:
        self.device = device
        self.gw = gw
        self.gain = gain

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']
        if self.device == 'cpu' and self.gw:
            r = img_frame[:, :, 0] \
                * (np.mean(img_frame[:, :, 1]) / np.mean(img_frame[:, :, 0]))
            g = img_frame[:, :, 1]
            b = img_frame[:, :, 2] \
                * (np.mean(img_frame[:, :, 1]) / np.mean(img_frame[:, :, 2]))
            img_frame = np.stack([r, g, b], axis=2)
            img_frame = np.clip(img_frame, 0, 1)
        elif self.device == 'cuda' and self.gw:
            r = img_frame[:, 0, :, :] * (torch.mean(img_frame[:, 1, :, :])
                                         / torch.mean(img_frame[:, 0, :, :]))
            g = img_frame[:, 1, :, :]
            b = img_frame[:, 2, :, :] * (torch.mean(img_frame[:, 1, :, :])
                                         / torch.mean(img_frame[:, 2, :, :]))
            img_frame = torch.stack([r, g, b], dim=1).clamp(0, 1)
        elif self.device == 'cpu' and not self.gw:
            r = img_frame[:, :, 0] * self.gain['r']
            g = img_frame[:, :, 1] * self.gain['g']
            b = img_frame[:, :, 2] * self.gain['b']
            img_frame = np.stack([r, g, b], axis=2)
            img_frame = np.clip(img_frame, 0, 1)
        elif self.device == 'cuda' and not self.gw:
            r = img_frame[:, :, 0] * self.gain['r']
            g = img_frame[:, :, 1] * self.gain['g']
            b = img_frame[:, :, 2] * self.gain['b']
            img_frame = torch.stack([r, g, b], dim=1).clamp(0, 1)
        else:
            raise NotImplementedError
        results['img_frame'] = img_frame
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class ColorCorrectionMatrix(BaseTransform):
    def __init__(self, ccm: list, device: str) -> None:
        self.ccm = ccm
        self.device = device

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']
        if self.device == 'cpu':
            r = img_frame[:, :, 0]
            g = img_frame[:, :, 1]
            b = img_frame[:, :, 2]
            r_ccm = float(self.ccm[0][0]) * r \
                + float(self.ccm[0][1]) * g + float(self.ccm[0][2]) * b
            g_ccm = float(self.ccm[1][0]) * r \
                + float(self.ccm[1][1]) * g + float(self.ccm[1][2]) * b
            b_ccm = float(self.ccm[2][0]) * r \
                + float(self.ccm[2][1]) * g + float(self.ccm[2][2]) * b
            img_frame = np.stack([r_ccm, g_ccm, b_ccm], axis=2)
            img_frame = np.clip(img_frame, 0, 1)
        elif self.device == 'cuda':
            r = img_frame[:, 0, :, :]
            g = img_frame[:, 1, :, :]
            b = img_frame[:, 2, :, :]
            r_ccm = float(self.ccm[0][0]) * r \
                + float(self.ccm[0][1]) * g + float(self.ccm[0][2]) * b
            g_ccm = float(self.ccm[1][0]) * r \
                + float(self.ccm[1][1]) * g + float(self.ccm[1][2]) * b
            b_ccm = float(self.ccm[2][0]) * r \
                + float(self.ccm[2][1]) * g + float(self.ccm[2][2]) * b
            img_frame = torch.stack([r_ccm, g_ccm, b_ccm], dim=1).clamp(0, 1)
        else:
            raise NotImplementedError
        results['img_frame'] = img_frame
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(ccm={self.ccm})'
        return repr_str


@TRANSFORMS.register_module()
class GammaCorrection(BaseTransform):
    def __init__(self,
                 digital_gain: float,
                 gamma: float,
                 device: str) -> None:
        self.digital_gain = digital_gain
        self.gamma = gamma
        self.device = device

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']
        if self.device == 'cpu':
            # digital gain
            img_frame = img_frame \
                * (float(self.digital_gain) / np.mean(img_frame))
            img_frame = np.clip(img_frame, 0, 1)
            # gamma correction
            img_frame = img_frame ** (1 / float(self.gamma))
            img_frame = np.clip(img_frame, 0, 1)
        elif self.device == 'cuda':
            # digital gain
            img_frame = img_frame \
                * (float(self.digital_gain) / torch.mean(img_frame))\
                .clamp(0, 1)
            # gamma correction
            img_frame = img_frame ** (1 / float(self.gamma))
            img_frame = img_frame.clamp(0, 1)
        else:
            raise NotImplementedError
        results['img_frame'] = img_frame
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(digital_gain={self.digital_gain}, '
        repr_str += f'gamma={self.gamma})'
        return repr_str


@TRANSFORMS.register_module()
class Torch2Cv2Image(BaseTransform):
    def __init__(self) -> None:
        pass

    def transform(self, results: dict) -> dict:
        bgr_image = results['img_frame']
        bgr_image = bgr_image.mul_(255).squeeze(0).permute(1, 2, 0)\
            .detach().byte().cpu().numpy()
        results['img_frame'] = bgr_image
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadCv2Image(BaseTransform):
    def __init__(self) -> None:
        pass

    def transform(self, results: dict) -> dict:
        img_path = results['img_path']
        if img_path.isalpha():
            img_frame = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        else:
            img_array = np.fromfile(img_path, np.uint8)
            img_frame = cv2.cvtColor(
                cv2.imdecode(img_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        results['img_frame'] = img_frame
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class ExecuteFastOpenISP(BaseTransform):
    def __init__(self, isp) -> None:
        self.isp = isp

    def transform(self, results: dict) -> dict:
        data, _ = self.isp.execute(results['img_frame'])
        results['img_frame'] = data['output']
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadCv2Frame(BaseTransform):
    def __init__(self, input_writer) -> None:
        self.input_writer = input_writer

    def transform(self, results: dict) -> dict:
        img_path = results['img_path']
        if img_path.isalpha():
            img_frame = cv2.imread(img_path)
        else:
            img_array = np.fromfile(img_path, np.uint8)
            img_frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BGR2RGB)
        self.input_writer.append_data(img_frame)
        results['img_frame'] = img_frame
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class StackFrames(BaseTransform):
    def __init__(self, stack_frames, num_frames, device) -> None:
        self.stack_frames = stack_frames
        self.num_frames = num_frames
        self.device = device

    def transform(self, results: dict) -> dict:
        if self.device == 'cpu':
            frame = np.expand_dims(results['img_frame'], axis=0)\
                .astype(np.float64)
            self.stack_frames = self.stack_frames \
                if self.stack_frames is not None else frame
            self.stack_frames = self.stack_frames[1:, :, :, :] \
                if self.stack_frames.shape[0] > 1 else self.stack_frames
            while self.stack_frames.shape[0] < self.num_frames:
                self.stack_frames = np.concatenate(
                    [self.stack_frames, frame], axis=0)
            results['stack_frames'] = self.stack_frames
        elif self.device == 'cuda':
            frame = torch.from_numpy(
                results['img_frame']).to(
                self.device).permute(2, 0, 1).unsqueeze(0).to(torch.float64)
            results['img_frame'] = frame
            self.stack_frames = self.stack_frames \
                if self.stack_frames is not None else frame
            self.stack_frames = self.stack_frames[1:, :, :, :] \
                if self.stack_frames.shape[0] > 1 else self.stack_frames
            while self.stack_frames.shape[0] < self.num_frames:
                self.stack_frames = torch.cat(
                    [self.stack_frames, frame], dim=0)
            results['stack_frames'] = self.stack_frames
        else:
            raise NotImplementedError
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class DeAfterImage(BaseTransform):
    def __init__(self, diff_frames, threshold, device) -> None:
        self.diff_frames = diff_frames
        self.threshold = threshold
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epsilon = 1e-3

    def transform(self, results: dict) -> dict:
        if self.device == 'cpu':
            self.mean_frames = np.mean(results['stack_frames'], axis=0)
            frames = results['stack_frames']

            for i in range(frames.shape[0] - 1):
                if self.diff_frames is not None and i != frames.shape[0] - 2:
                    continue
                else:
                    diff_per_frm = np.abs(
                        frames[i + 1, :, :, :] - frames[i, :, :, :])
                    diff_per_frm /= (np.max(diff_per_frm) + self.epsilon)
                    diff_per_frm = np.where(
                        diff_per_frm > float(self.threshold), 1, 0)
                    self.diff_frames = np.logical_or(
                        self.diff_frames, diff_per_frm) \
                        if self.diff_frames is not None else diff_per_frm

            results['img_frame'] = np.where(self.diff_frames == 1,
                                            results['img_frame'],
                                            self.mean_frames) \
                if self.diff_frames is not None else self.mean_frames
            results['img_frame'] = results['img_frame'].astype(np.uint8)
        elif self.device == 'cuda':
            self.mean_frames = torch.mean(results['stack_frames'], dim=0)

            frames = results['stack_frames']
            for i in range(frames.shape[0] - 1):
                if self.diff_frames is not None and i != frames.shape[0] - 2:
                    continue
                else:
                    diff_per_frm = torch.abs(
                        frames[i + 1:, :, :, :] - frames[i, :, :, :])
                    diff_per_frm /= (torch.max(diff_per_frm) + self.epsilon)
                    diff_per_frm = torch.where(
                        diff_per_frm > float(self.threshold), 1, 0)
                    self.diff_frames = torch.logical_or(
                        self.diff_frames, diff_per_frm) \
                        if self.diff_frames is not None else diff_per_frm

            results['img_frame'] = torch.where(self.diff_frames == 1,
                                               results['img_frame'],
                                               self.mean_frames)\
                if self.diff_frames is not None else self.mean_frames
            results['img_frame'] = results['img_frame']\
                .squeeze(0).permute(1, 2, 0).detach().cpu().byte().numpy()
        else:
            raise NotImplementedError
        results['diff_frames'] = self.diff_frames
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class FrameFilter(BaseTransform):
    def __init__(self, config) -> None:
        self.cfg = config

    def transform(self, results: dict) -> dict:
        YUV = cv2.split(cv2.cvtColor(results['img_frame'], cv2.COLOR_RGB2YUV))
        new = []
        for c in range(len(YUV)):
            _channel = YUV[c]
            luma_or_chroma = int(c != 0)    # luma -> 0, chroma -> 1

            if self.cfg['nr3d.eeh'] and not luma_or_chroma:
                ksize = int(self.cfg['nr3d.eeh_ksize'])
                edge = float(self.cfg['nr3d.eeh_edge_factor'])
                sign = float(self.cfg['nr3d.eeh_signal_factor'])
                sharpening_filter = []
                for f in range(ksize):
                    sharpening_filter.append(
                        [-edge] * (ksize//2)
                        + [(ksize**2 - 1) * edge + sign]
                        + [-edge] * (ksize//2)) \
                        if f == ksize//2 else sharpening_filter.append(
                        [-edge] * ksize)
                sharpening_filter = np.array(sharpening_filter)
                _channel = cv2.filter2D(_channel, -1, sharpening_filter)

            if self.cfg['nr3d.lpf_luma'] \
                    and self.cfg['nr3d.lpf_luma_bnf'] \
                    and not luma_or_chroma:
                new.append(
                    cv2.bilateralFilter(
                        _channel,
                        int(self.cfg['nr3d.lpf_luma_ksize']),
                        int(self.cfg['nr3d.lpf_luma_bnf_sigma.color']),
                        int(self.cfg['nr3d.lpf_luma_bnf_sigma.space'])))
            elif self.cfg['nr3d.lpf_luma'] \
                    and self.cfg['nr3d.lpf_luma_median'] \
                    and not luma_or_chroma:
                new.append(cv2.medianBlur(
                    _channel,
                    int(self.cfg['nr3d.lpf_luma_ksize'])))
            elif self.cfg['nr3d.lpf_chroma'] \
                    and self.cfg['nr3d.lpf_chroma_median'] \
                    and luma_or_chroma:
                new.append(cv2.medianBlur(
                    _channel,
                    int(self.cfg['nr3d.lpf_chroma_ksize'])))
            else:
                new.append(_channel)

        results['img_frame'] = cv2.cvtColor(cv2.merge(new), cv2.COLOR_YUV2RGB)
        # print(f'{self.__repr__()} Completed!')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class CropCv2Image(BaseTransform):
    def __init__(self, roi, window) -> None:
        self.refPt = [] if roi is None else roi
        self.window = window

    def transform(self, results: dict) -> dict:
        img_frame = results['img_frame']

        if len(self.refPt) == 2:
            roi = img_frame[self.refPt[0][1]*2:self.refPt[1][1]*2,
                            self.refPt[0][0]*2:self.refPt[1][0]*2]
        elif len(self.refPt) == 0:
            clone = img_frame.copy()
            img_interp = cv2.resize(img_frame, dsize=(0, 0), fx=0.5, fy=0.5)
            cv2.namedWindow("Image to Crop")
            cv2.setMouseCallback("Image to Crop", self.click_and_crop,
                                 param=img_interp)

            while True:
                cv2.imshow("Image to Crop", img_interp)
                key = cv2.waitKey(1) & 0xFF

                if len(self.refPt) == 2:
                    roi = clone[self.refPt[0][1]*2:self.refPt[1][1]*2,
                                self.refPt[0][0]*2:self.refPt[1][0]*2]

                if key == ord("r"):
                    os.execl(sys.executable, sys.executable, *sys.argv)
                elif key == ord("c"):
                    cv2.imshow("ROI", roi)
                elif key == ord("q"):
                    cv2.destroyAllWindows()
                    break
        else:
            raise NotImplementedError

        results['img_frame'] = roi
        results['img_roi'] = self.refPt
        # print(f'{self.__repr__()} Completed!')
        return results

    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x, y))
            cv2.rectangle(param,
                          self.refPt[0], self.refPt[1], (0, 255, 0), 2)
            cv2.imshow("Image to Crop", param)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
