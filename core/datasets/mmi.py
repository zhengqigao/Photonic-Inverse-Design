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

resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

__all__ = ["MMI", "MMIDataset"]


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
        
        noise = gen_gaussian_noise(
            data[:,:, 1:2], self.mean, self.std, random_state=self.random_state
        )
        # print(noise)
        # noise needs to be masked, however, input_len is not stored. We just approximate the input_len as 1/7 the total length
        noise[..., int(noise.shape[-2]/7):, :].fill_(0)
        data = data.clone()
        data[:,:, 1:2] += noise
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
        mode: str = "bicubic",
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
        x = data.reshape(-1, 1, data.shape[-2], data.shape[-1])

        if self.scale_factor is not None:
            size = (
                int(data.shape[-2] * self.scale_factor),
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
                    size=data.shape[-2:],
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
                    size=data.shape[-2:],
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
                size=data.shape[-2:],
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
        mode: str = "bicubic",
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
        # target: [bs, #data, 2, h, w]
        if self.scale_factor == 1:
            return data, target
        x = target.reshape(-1, 1, target.shape[-2], target.shape[-1])

        if self.scale_factor is not None:
            size = (
                int(target.shape[-2] * self.scale_factor),
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
                    size=target.shape[-2:],
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
                    size=target.shape[-2:],
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
                size=target.shape[-2:],
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
        assert prec in set(["fp32", "fp64", "fp16", "bfp16", "int32", "int16"] + [f"int{i}" for i in range(1, 16)])

    def _quantize_int(self, data, k: int = 32):
        # per field scaling
        if self.prec in {"int1", "int2", "int3", "int4", "int5", "int6"}:
            _q = 0.015
        else:
            _q = 0
        if self.mode == "per_channel":
            v_max = torch.quantile(
                data.flatten(-2, -1), q=1 - _q, dim=-1, keepdim=True
            )[..., None]
            v_min = torch.quantile(data.flatten(-2, -1), q=_q, dim=-1, keepdim=True)[
                ..., None
            ]

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
        assert prec in set(["fp32", "fp64", "fp16", "bfp16", "int32", "int16"] + [f"int{i}" for i in range(1, 16)])

    def _quantize_int(self, data, k: int = 32):
        # per field scaling
        if self.prec in {"int1", "int2", "int3", "int4", "int5", "int6"}:
            _q = 0.015
        else:
            _q = 0
        if self.mode == "per_channel":
            v_max = torch.quantile(
                data.flatten(-2, -1), q=1 - _q, dim=-1, keepdim=True
            )[..., None]
            v_min = torch.quantile(data.flatten(-2, -1), q=_q, dim=-1, keepdim=True)[
                ..., None
            ]

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


class MMI(VisionDataset):
    url = None
    filename_suffix = "fields_epsilon_mode.pt"
    train_filename = "training"
    test_filename = "test"
    folder = "mmi"

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
        train_ratio: float = 0.7,
        pol_list: List[str] = ["Hz"],
        processed_dir: str = "processed",
        download: bool = False,
        noise_cfg=dict(),
    ) -> None:
        self.processed_dir = processed_dir
        self.noise_cfg = noise_cfg
        root = os.path.join(os.path.expanduser(root), self.folder)
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.train_ratio = train_ratio
        self.pol_list = sorted(pol_list)
        self.filenames = [f"{pol}_{self.filename_suffix}" for pol in self.pol_list]
        self.train_filename = self.train_filename
        self.test_filename = self.test_filename

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.wavelength: Any = []
        self.grid_step: Any = []
        self.data: Any = []
        self.targets = []
        self.eps_min = self.eps_max = None

        self.process_raw_data()
        self.wavelength, self.grid_step, self.data, self.targets = self.load(
            train=train
        )
        self.data, self.targets = self.add_noise(
            self.data, self.targets, self.noise_cfg
        )
        self.data = self.data.to(torch.complex32)
        self.targets = self.targets.to(torch.complex32)

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
                wavelength, grid_step, data, targets, _, _ = torch.load(f)
                if data.shape[0:2] == targets.shape[0:2]:
                    print("Data already processed")
                    return
        wavelength, grid_step, data, targets, eps_min, eps_max = self._load_dataset()
        (
            wavelength_train,
            grid_step_train,
            data_train,
            targets_train,
            wavelength_test,
            grid_step_test,
            data_test,
            targets_test,
        ) = self._split_dataset(wavelength, grid_step, data, targets)
        data_train, data_test = self._preprocess_dataset(data_train, data_test)
        self._save_dataset(
            wavelength_train,
            grid_step_train,
            data_train,
            targets_train,
            wavelength_test,
            grid_step_test,
            data_test,
            targets_test,
            eps_min,
            eps_max,
            processed_dir,
            self.train_filename,
            self.test_filename,
        )

    def _load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_data = [
            torch.load(os.path.join(self.root, "raw", filename))
            for filename in self.filenames
        ]
        data = torch.cat(
            [rd["eps"] for rd in raw_data], dim=0
        )  # use Ex Ey Ez fields for now
        wavelength = torch.cat([rd["wavelength"] for rd in raw_data], dim=0)
        grid_step = torch.cat([rd["grid_step"] for rd in raw_data], dim=0)
        targets = torch.cat(
            [rd["fields"] for rd in raw_data], dim=0
        )  # use Ex Ey Ez field for now
        self.eps_min = raw_data[0]["eps_min"]
        self.eps_max = raw_data[0]["eps_max"]

        return wavelength, grid_step, data, targets, self.eps_min, self.eps_max

    def _split_dataset(
        self, wavelength: Tensor, grid_step: Tensor, data: Tensor, targets: Tensor
    ) -> Tuple[Tensor, ...]:
        from sklearn.model_selection import train_test_split

        (
            wavelength_train,
            wavelength_test,
            grid_step_train,
            grid_step_test,
            data_train,
            data_test,
            targets_train,
            targets_test,
        ) = train_test_split(
            wavelength,
            grid_step,
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
            wavelength_train,
            grid_step_train,
            data_train,
            targets_train,
            wavelength_test,
            grid_step_test,
            data_test,
            targets_test,
        )

    def _preprocess_dataset(
        self, data_train: Tensor, data_test: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return data_train, data_test

    @staticmethod
    def _save_dataset(
        wavelength_train: Tensor,
        grid_step_train: Tensor,
        data_train: Tensor,
        targets_train: Tensor,
        wavelength_test: Tensor,
        grid_step_test: Tensor,
        data_test: Tensor,
        targets_test: Tensor,
        eps_min: Tensor,
        eps_max: Tensor,
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
                    wavelength_train,
                    grid_step_train,
                    data_train,
                    targets_train,
                    eps_min,
                    eps_max,
                ),
                f,
            )

        with open(processed_test_file, "wb") as f:
            torch.save(
                (
                    wavelength_test,
                    grid_step_test,
                    data_test,
                    targets_test,
                    eps_min,
                    eps_max,
                ),
                f,
            )
        print(f"Processed dataset saved")

    def load(self, train: bool = True):
        filename = f"{self.train_filename}.pt" if train else f"{self.test_filename}.pt"
        with open(os.path.join(self.root, self.processed_dir, filename), "rb") as f:
            (
                wavelength,
                grid_step,
                data,
                targets,
                self.eps_min,
                self.eps_max,
            ) = torch.load(f)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
            if isinstance(wavelength, np.ndarray):
                wavelength = torch.from_numpy(wavelength)
            if isinstance(grid_step, np.ndarray):
                grid_step = torch.from_numpy(grid_step)
        return wavelength, grid_step, data, targets

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

    def _check_integrity(self) -> bool:
        return all(
            [
                os.path.exists(os.path.join(self.root, "raw", filename))
                for filename in self.filenames
            ]
        )

    def __len__(self):
        return self.targets.size(0)

    def __getitem__(self, item):
        # wave [bs, 2, 1] real
        # data [bs, 2, 6, h, w] real # 224x224 -> 80x520, 96x384
        # bs = #Wavelength * #epsilon combinatio
        # N = #ports/wavelenth/epsilon combination
        # data [bs, 2, 6, H, W] -> [bs, 2, 4, H, W]
        # target [bs, 2, 4, h, w] real grid_step ->
        return (
            self.wavelength[item],
            self.grid_step[item],
            self.data[item].to(torch.cfloat),
            self.targets[item].to(torch.cfloat),
        )

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class MMIDataset:
    def __init__(
        self,
        root: str,
        split: str,
        test_ratio: float,
        train_valid_split_ratio: List[float],
        pol_list: List[str],
        processed_dir: str = "processed",
        train_noise_cfg=dict(),
    ):
        self.root = root
        self.split = split
        self.test_ratio = test_ratio
        assert 0 < test_ratio < 1, print(
            f"Only support test_ratio from (0, 1), but got {test_ratio}"
        )
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.eps_min = None
        self.eps_max = None
        self.pol_list = sorted(pol_list)
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
            train_valid = MMI(
                self.root,
                train=True,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                pol_list=self.pol_list,
                processed_dir=self.processed_dir,
                noise_cfg=self.train_noise_cfg,
            )
            self.eps_min = train_valid.eps_min
            self.eps_max = train_valid.eps_max

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            if (
                self.train_valid_split_ratio[0] + self.train_valid_split_ratio[1]
                > 0.99999
            ):
                valid_len = len(train_valid) - train_len
            else:
                valid_len = int(self.train_valid_split_ratio[1] * len(train_valid))
                train_valid.data = train_valid.data[: train_len + valid_len]
                train_valid.wavelength = train_valid.wavelength[: train_len + valid_len]
                train_valid.grid_step = train_valid.grid_step[: train_len + valid_len]
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
            test = MMI(
                self.root,
                train=False,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                pol_list=self.pol_list,
                processed_dir=self.processed_dir,
            )

            self.data = test
            self.eps_min = test.eps_min
            self.eps_max = test.eps_max

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, index: int) -> Dict[str, Tensor]:
        return self.__getitem__(index)


def test_mmi():
    import pdb

    # pdb.set_trace()
    mmi = MMI(root="../../data", download=True, processed_dir="random_size5")
    print(mmi.data.size(), mmi.targets.size())
    mmi = MMI(
        root="../../data", train=False, download=True, processed_dir="random_size5"
    )
    print(mmi.data.size(), mmi.targets.size())
    mmi = MMIDataset(
        root="../../data",
        split="train",
        test_ratio=0.1,
        train_valid_split_ratio=[0.9, 0.1],
        pol_list=[f"rHz_{i}" for i in range(5)],
    )
    print(len(mmi))


if __name__ == "__main__":
    test_mmi()
