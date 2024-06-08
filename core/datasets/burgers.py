"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-24 23:27:31
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-01 16:22:32
"""
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-04 13:38:43
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-04-04 15:16:55
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from pyutils.compute import add_gaussian_noise, gen_gaussian_noise
from pyutils.general import print_stat
from torch import Tensor
from torchpack.datasets.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url
from torchvision.transforms import InterpolationMode
from pathlib import Path
from neuralop.utils import UnitGaussianNormalizer
import scipy

resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

__all__ = ["Burgers", "BurgersDataset"]


class InputGaussianError(object):
    def __init__(
        self, mean: float = 0.0, std: float = 0.0, random_state: int = 0
    ) -> None:
        self.mean = mean
        self.std = std
        self.random_state = random_state

    def __call__(self, data, target):
        # random_state make sures we can reproduce the noises
        # we only added to the source field (channel 1 in the 3rd dim), not the permittivity
        if self.mean < 1e-6 and self.std < 1e-6:
            return data, target
        if data.is_complex():
            is_complex = True
            data = torch.view_as_real(data)
        else:
            is_complex = False

        data = add_gaussian_noise(
            data, self.mean, self.std, random_state=self.random_state
        )
        if is_complex:
            data = torch.view_as_complex(data)
        return data, target


class TargetGaussianError(object):
    def __init__(
        self, mean: float = 0.0, std: float = 0.0, random_state: int = 0
    ) -> None:
        self.mean = mean
        self.std = std
        self.random_state = random_state

    def __call__(self, data, target):
        # random_state make sures we can reproduce the noises
        if self.mean < 1e-6 and self.std < 1e-6:
            return data, target
        if target.is_complex():
            is_complex = True
            target = torch.view_as_real(target)
        else:
            is_complex = False
        target = add_gaussian_noise(
            target, self.mean, self.std, random_state=self.random_state
        )
        if is_complex:
            target = torch.view_as_complex(target)
        return data, target


class InputDownSampleError(object):
    def __init__(
        self,
        size: None | Tuple[int, int] = None,
        scale_factor: float | Tuple | None = None,
        mode: str = "linear",
        antialias: bool = False,
        align_corners: bool = True,
    ) -> None:
        self.size = size  # [pixel dimension, not physical dimension]
        self.scale_factor = scale_factor
        if scale_factor is not None:
            self.size = None
        self.mode = mode
        self.antialias = antialias
        self.align_corners = align_corners

    def __call__(self, data, target):
        # epsilon source will all have downsampling error
        # data: [bs, #data, 2, h, w]
        if self.scale_factor == 1:
            return data, target
        x = data.reshape(-1, 1, data.shape[-1])

        if self.scale_factor is not None:
            size = (
                int(data.shape[-1] * self.scale_factor),
            )
        if x.is_complex():
            x = torch.complex(
                torch.nn.functional.interpolate(
                    torch.nn.functional.interpolate(
                        x.real,
                        size=size,
                        mode=self.mode,
                        antialias=self.antialias,
                        align_corners=self.align_corners,
                    ),
                    size=data.shape[-1],
                    mode=self.mode,
                    antialias=self.antialias,
                    align_corners=self.align_corners,
                ),
                torch.nn.functional.interpolate(
                    torch.nn.functional.interpolate(
                        x.imag,
                        size=size,
                        mode=self.mode,
                        antialias=self.antialias,
                        align_corners=self.align_corners,
                    ),
                    size=data.shape[-1],
                    mode=self.mode,
                    antialias=self.antialias,
                    align_corners=self.align_corners,
                ),
            )
        else:
            x = torch.nn.functional.interpolate(
                torch.nn.functional.interpolate(
                    x,
                    size=size,
                    mode=self.mode,
                    antialias=self.antialias,
                    align_corners=self.align_corners,
                ),
                size=data.shape[-1],
                mode=self.mode,
                antialias=self.antialias,
                align_corners=self.align_corners,
            )
        data = x.view_as(data)

        return data, target


class TargetDownSampleError(object):
    def __init__(
        self,
        size: None | Tuple[int, int] = None,
        scale_factor: float | Tuple | None = None,
        mode: str = "linear",
        antialias: bool = False,
        align_corners: bool = True,
    ) -> None:
        self.size = size  # [pixel dimension, not physical dimension]
        self.scale_factor = scale_factor
        if scale_factor is not None:
            self.size = None
        self.mode = mode
        self.antialias = antialias
        self.align_corners = align_corners

    def __call__(self, data, target):
        # epsilon source will all have downsampling error
        # target: [bs, #data, 2, h]
        if self.scale_factor == 1:
            return data, target
        x = target.reshape(-1, 1, target.shape[-1])

        if self.scale_factor is not None:
            size = (
                int(target.shape[-1] * self.scale_factor),
            )
        if x.is_complex():
            x = torch.complex(
                torch.nn.functional.interpolate(
                    torch.nn.functional.interpolate(
                        x.real,
                        size=size,
                        mode=self.mode,
                        antialias=self.antialias,
                        align_corners=self.align_corners,
                    ),
                    size=target.shape[-1],
                    mode=self.mode,
                    antialias=self.antialias,
                    align_corners=self.align_corners,
                ),
                torch.nn.functional.interpolate(
                    torch.nn.functional.interpolate(
                        x.imag,
                        size=size,
                        mode=self.mode,
                        antialias=self.antialias,
                        align_corners=self.align_corners,
                    ),
                    size=target.shape[-1],
                    mode=self.mode,
                    antialias=self.antialias,
                    align_corners=self.align_corners,
                ),
            )
        else:
            x = torch.nn.functional.interpolate(
                torch.nn.functional.interpolate(
                    x,
                    size=size,
                    mode=self.mode,
                    antialias=self.antialias,
                    align_corners=self.align_corners,
                ),
                size=target.shape[-1],
                mode=self.mode,
                antialias=self.antialias,
                align_corners=self.align_corners,
            )
        target = x.view_as(target)

        return data, target


class InputQuantizationError(object):
    def __init__(self, prec: str = "fp32", mode="per_channel") -> None:
        # emulate the quantization error
        # the target precision is for real or imag part
        # orig prec -> convert to assigned prec -> convert it to fp32
        self.prec = prec
        self.mode = mode
        assert mode in {"per_channel", "per_tensor"}
        # assert prec in {"fp32", "fp64", "fp16", "int32", "int16", "int8", "int4"}
        assert prec in set(
            ["fp32", "fp64", "fp16", "bfp16", "int32", "int16"]
            + [f"int{i}" for i in range(1, 16)]
        )

    def _quantize_int(self, data, k: int = 32):
        # per field scaling
        if self.prec in {"int1", "int2", "int3", "int4", "int5", "int6"}:
            _q = 0.015
        else:
            _q = 0
        if self.mode == "per_channel":
            v_max = torch.quantile(
                data, q=1 - _q, dim=-1, keepdim=True
            )
            v_min = torch.quantile(data, q=_q, dim=-1, keepdim=True)

        elif self.mode == "per_tensor":
            v_max = torch.quantile(data, q=1 - _q)
            v_min = torch.quantile(data, q=_q)
        data = data.clamp(v_min, v_max)
        scale = v_max - v_min
        # data = data / (2 * scale + 1e-10) + 0.5  # [0, 1]
        data = (data - v_min) / (scale + 1e-10)
        n = float(2**k - 1)
        out = torch.round(data * n) / (n / (scale + 1e-10)) + v_min
        # out = (out - 0.5) * (2 * scale + 1e-10)  # [-scale, scale] the original range
        return out

    def __call__(self, data: Tensor, target: Tensor):
        # assume input is complex64 or complex128
        is_complex = False
        if data.is_complex():
            is_complex = True
            data = torch.view_as_real(data)
        if self.prec == "fp64":
            data = data.to(torch.double).to(torch.float32)
        elif self.prec == "fp32":
            data = data.to(torch.float32)
        elif self.prec == "fp16":
            data = data.half().to(torch.float32)
        elif self.prec == "bfp16":
            data = data.to(torch.bfloat16).to(torch.float32)
        elif self.prec.startswith("int"):
            data = self._quantize_int(data, k=int(self.prec[3:]))
        else:
            raise NotImplementedError

        if is_complex:
            data = torch.view_as_complex(data)
        return data, target


class TargetQuantizationError(object):
    def __init__(self, prec: str = "fp32", mode="per_channel") -> None:
        # emulate the quantization error
        # the target precision is for real or imag part
        # orig prec -> convert to assigned prec -> convert it to fp32
        self.prec = prec
        self.mode = mode
        assert mode in {"per_channel", "per_tensor"}
        assert prec in set(
            ["fp32", "fp64", "fp16", "bfp16", "int32", "int16"]
            + [f"int{i}" for i in range(1, 16)]
        )

    def _quantize_int(self, data, k: int = 32):
        # per field scaling
        if self.prec in {"int1", "int2", "int3", "int4", "int5", "int6"}:
            _q = 0.015
        else:
            _q = 0
        if self.mode == "per_channel":
            v_max = torch.quantile(
                data, q=1 - _q, dim=-1, keepdim=True
            )
            v_min = torch.quantile(data, q=_q, dim=-1, keepdim=True)

        elif self.mode == "per_tensor":
            v_max = torch.quantile(data, q=1 - _q)
            v_min = torch.quantile(data, q=_q)
        scale = v_max - v_min
        # data = data / (2 * scale + 1e-10) + 0.5  # [0, 1]
        data = (data - v_min) / (scale + 1e-10)
        n = float(2**k - 1)
        out = torch.round(data * n) / (n / (scale + 1e-10)) + v_min
        # out = (out - 0.5) * (2 * scale + 1e-10)  # [-scale, scale] the original range
        return out

    def __call__(self, data: Tensor, target: Tensor):
        # assume input is complex64 or complex128
        is_complex = False
        if target.is_complex():
            is_complex = True
            target = torch.view_as_real(target)
        if self.prec == "fp64":
            target = target.to(torch.double).to(torch.float32)
        elif self.prec == "fp32":
            target = target.to(torch.float32)
        elif self.prec == "fp16":
            target = target.half().to(torch.float32)
        elif self.prec == "bfp16":
            target = target.to(torch.bfloat16).to(torch.float32)
        elif self.prec.startswith("int"):
            target = self._quantize_int(target, k=int(self.prec[3:]))
        else:
            raise NotImplementedError

        if is_complex:
            target = torch.view_as_complex(target)
        return data, target


class Burgers(VisionDataset):
    url = None
    train_filename = "training"
    test_filename = "test"
    folder = "burgers"

    noise_fn_list = {
        "input_gaussian": InputGaussianError,
        "target_gaussian": TargetGaussianError,
        "input_downsample": InputDownSampleError,
        "target_downsample": TargetDownSampleError,
        "input_quant": InputQuantizationError,
        "target_quant": TargetQuantizationError,
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.9,
        ssr: int = 1,
        append_pos: bool = True,
        encode_input: bool=True,
        encode_output: bool=True,
        processed_dir: str = "processed",
        download: bool = False,
        noise_cfg=dict(),
    ) -> None:
        self.processed_dir = processed_dir
        self.noise_cfg = noise_cfg
        self.append_pos = append_pos
        self.ssr = ssr
        root = os.path.join(os.path.expanduser(root), self.folder)
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.train_ratio = train_ratio
        self.encode_input = encode_input
        self.encode_output = encode_output
        self.filename = "burgers_data_R10.mat"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.data: Any = []
        self.targets = []

        self.process_raw_data()
        self.data, self.targets = self.load(train=train)
        self.data, self.targets = self.add_noise(
            self.data, self.targets, self.noise_cfg
        )

        B, T, X = self.data.shape
        self.grid_x = None
        if self.append_pos:
            # Note that linspace is inclusive of both ends
            self.grid_x = torch.linspace(0, 1, X+1)[:-1].view(1, -1)

        self.data = self.data
        self.targets = self.targets

    def add_noise(self, data, target, noise_cfg):
        print(f"Adding noise: {noise_cfg}")
        for noise_type, noise_arg in noise_cfg.items():
            noise_fn = self.noise_fn_list[noise_type](**noise_arg)
            data, target = noise_fn(data, target)
        return data, target

    def process_raw_data(self) -> None:
        processed_dir = os.path.join(self.root, self.processed_dir)
        processed_training_file = os.path.join(
            processed_dir, f"{self.train_filename}.pt"
        )
        processed_test_file = os.path.join(processed_dir, f"{self.test_filename}.pt")
        if os.path.exists(processed_training_file) and os.path.exists(
            processed_test_file
        ):
            with open(
                os.path.join(self.root, self.processed_dir, f"{self.test_filename}.pt"),
                "rb",
            ) as f:
                data, targets = torch.load(f)
                if data.shape[0:2] == targets.shape[0:2]:
                    print("Data already processed")
                    return
        data, targets = self._load_dataset()
        data_train, targets_train, data_test, targets_test = self._split_dataset(
            data, targets
        )
        data_train, targets_train, data_test, targets_test = self._preprocess_dataset(
            data_train, targets_train, data_test, targets_test
        )  # this can already split train and test using self.train

        self._save_dataset(
            data_train,
            targets_train,
            data_test,
            targets_test,
            processed_dir=processed_dir,
        )

    def _load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        file = f"raw/{self.filename}"
        filepath = Path(self.root).joinpath(file).as_posix()
        raw_data = scipy.io.loadmat(filepath)
        data = torch.from_numpy(raw_data["a"]).reshape(-1, 1, 8192).float()
        targets = torch.from_numpy(raw_data["u"]).reshape(-1, 1, 8192).float()
        del raw_data

        return data, targets

    def _split_dataset(self, data: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        from sklearn.model_selection import train_test_split

        (
            data_train,
            data_test,
            targets_train,
            targets_test,
        ) = train_test_split(
            data,
            targets,
            train_size=self.train_ratio,
            random_state=42,
        )
        print(
            f"training: {data_train.shape[0]} examples, "
            f"test: {data_test.shape[0]} examples"
        )
        return (
            data_train,
            targets_train,
            data_test,
            targets_test,
        )

    def _preprocess_dataset(
        self, train_data, train_targets, test_data, test_targets
    ) -> Tuple[Tensor, Tensor]:
        if self.encode_input:
            reduce_dims = list(range(train_data.ndim))

            input_encoder = UnitGaussianNormalizer(train_data, reduce_dim=reduce_dims)
            train_data = input_encoder.encode(train_data)
            test_data = input_encoder.encode(test_data)
        else:
            input_encoder = None

        if self.encode_output and self.train:  # no encodeing for test
            reduce_dims = list(range(train_targets.ndim))

            output_encoder = UnitGaussianNormalizer(
                train_targets, reduce_dim=reduce_dims
            )
            train_targets = output_encoder.encode(train_targets)
            test_targets = output_encoder.encode(test_targets)
        else:
            output_encoder = None

        return train_data, train_targets, test_data, test_targets

    @staticmethod
    def _save_dataset(
        data_train: Tensor,
        targets_train: Tensor,
        data_test: Tensor,
        targets_test: Tensor,
        processed_dir: str,
        train_filename: str = "training",
        test_filename: str = "test",
    ) -> None:
        os.makedirs(processed_dir, exist_ok=True)
        processed_training_file = os.path.join(processed_dir, f"{train_filename}.pt")
        processed_test_file = os.path.join(processed_dir, f"{test_filename}.pt")
        with open(processed_training_file, "wb") as f:
            torch.save(
                (
                    data_train,
                    targets_train,
                ),
                f,
            )

        with open(processed_test_file, "wb") as f:
            torch.save(
                (
                    data_test,
                    targets_test,
                ),
                f,
            )
        print(f"Processed dataset saved")

    def load(self, train: bool = True):
        filename = f"{self.train_filename}.pt" if train else f"{self.test_filename}.pt"
        with open(os.path.join(self.root, self.processed_dir, filename), "rb") as f:
            (
                data,
                targets,
            ) = torch.load(f)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)

        return data[..., ::self.ssr].contiguous(), targets[..., ::self.ssr].contiguous()

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

    def _check_integrity(self) -> bool:
        return all(
            [
                os.path.exists(os.path.join(self.root, "raw", self.filename))
            ]
        )

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, item):
        if self.grid_x is not None:
            data = torch.cat([self.data[item], self.grid_x], dim=0)
        else:
            data = self.data[item]
        return (data, self.targets[item])

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class BurgersDataset:
    def __init__(
        self,
        root: str,
        split: str,
        train_valid_split_ratio: List[float],
        test_ratio: float = 0.1,
        ssr: int = 1,
        encode_input: bool=True,
        encode_output: bool=True,
        processed_dir: str = "processed",
        train_noise_cfg=dict(),
    ):
        self.root = root
        self.split = split

        self.test_ratio = test_ratio
        self.train_valid_split_ratio = train_valid_split_ratio
        self.ssr = ssr
        self.encode_input = encode_input
        self.encode_output = encode_output
        self.data = None

        self.processed_dir = processed_dir
        self.train_noise_cfg = train_noise_cfg

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        tran = [
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            train_valid = Burgers(
                self.root,
                train=True,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                ssr=self.ssr,
                encode_input=self.encode_input,
                encode_output=self.encode_output,
                processed_dir=self.processed_dir,
                noise_cfg=self.train_noise_cfg,
            )

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            if (
                self.train_valid_split_ratio[0] + self.train_valid_split_ratio[1]
                > 0.99999
            ):
                valid_len = len(train_valid) - train_len
            else:
                valid_len = int(self.train_valid_split_ratio[1] * len(train_valid))
                train_valid.data = train_valid.data[: train_len + valid_len]
                train_valid.targets = train_valid.targets[: train_len + valid_len]

            split = [train_len, valid_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )

            if self.split == "train":
                self.data = train_subset
            else:
                self.data = valid_subset

        else:
            test = Burgers(
                self.root,
                train=False,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                ssr=self.ssr,
                encode_input=self.encode_input,
                encode_output=self.encode_output,
                processed_dir=self.processed_dir,
                noise_cfg=self.train_noise_cfg,
            )

            self.data = test

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, index: int) -> Dict[str, Tensor]:
        return self.__getitem__(index)


def test_darcy():
    import pdb

    # pdb.set_trace()
    mmi = Burgers(root="../../data", download=True)
    print(mmi.data.size(), mmi.targets.size())
    mmi = Burgers(root="../../data", train=False, download=True)
    print(mmi.data.size(), mmi.targets.size())
    mmi = BurgersDataset(
        root="../../data",
        split="train",
        train_valid_split_ratio=[0.9, 0.1],
    )
    print(len(mmi))


if __name__ == "__main__":
    test_darcy()
