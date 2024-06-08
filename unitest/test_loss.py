"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-19 00:38:30
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-01-25 02:07:26
"""
"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-19 00:17:20
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-01-19 00:17:20
"""
import numpy as np
import torch
from core.models import FNO2d, MaxwellSolver2D
from core.models.fno_cnn import EncoderDecoder
from core.utils import (
    CurlLoss,
    DivergenceLoss,
    IntensityMSELoss,
    FaradayResidue2d,
    ComplexL1Loss,
    ComplexMSELoss,
    ComplexTVLoss,
    PhaseLoss,
    TMPropagateFieldLoss,
)
import matplotlib.pyplot as plt
from core.models.simulator import Simulation2D
from core.models.constant import *


def test_loss():
    wavelength = 1.550
    device = torch.device("cuda:0")
    domain_size = (20, 100)
    grid_step = wavelength / 20
    domain_size_pixel = [round(i / grid_step) for i in domain_size]
    layer = FNO2d(
        in_channels=2,
        dim=8,
        kernel_list=[8],
        kernel_size_list=[1],
        padding_list=[0],
        hidden_list=[128],
        mode_list=[(4, 4)],
        act_func="GELU",
        domain_size=domain_size,
        grid_step=grid_step,
        device=device,
    ).to(device)
    x = torch.randn(1, 1, *domain_size_pixel, dtype=torch.cfloat, device=device)
    y = layer(x, wavelength=wavelength)
    faraday_loss = FaradayResidue2d(grid_step=grid_step, wavelength=wavelength)
    intensity_mse_loss = IntensityMSELoss()
    loss_pde = faraday_loss(y, x)
    target = torch.randn(1, 1, *domain_size_pixel, dtype=torch.float, device=device).square()
    loss_mse = intensity_mse_loss(y, target)
    loss = loss_pde + loss_mse
    print(loss.item(), loss_pde.item(), loss_mse.item())
    loss.backward()


def normalize(x):
    # x_mean = np.mean(x)
    # x_std = np.std(x)
    # if x_std > 1e-6:
    #     x = (x - x_mean) / x_std
    # else:
    #     x = x - x_mean
    # return x
    if isinstance(x, np.ndarray):
        x_min, x_max = np.percentile(x, 5), np.percentile(x, 95)
        x = np.clip(x, x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    else:
        x_min, x_max = torch.quantile(x, 0.05), torch.quantile(x, 0.95)
        x = x.clamp(x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    return x


def learn_field():
    import cv2

    device = torch.device("cuda:0")
    wavelength = 1.550
    device = torch.device("cuda:0")

    # img = torch.load("unitest/wave_1.pt").transpose(-1,-2)
    # img = img[0:1, -1, 10:-10, 10:-10]
    img = cv2.imread("unitest/e_field_1.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.resize(img, (192, 320))
    N = 100
    img = cv2.resize(img, (N, N))
    h, w = img.shape[0], img.shape[1]
    img = torch.from_numpy(img).to(device).unsqueeze(0).unsqueeze(0)
    h, w = img.shape[-2:]
    img = img.to(torch.cfloat)
    print(h, w)

    epsilon = cv2.imread("unitest/epsilon_field_1.png")
    epsilon = cv2.cvtColor(epsilon, cv2.COLOR_BGR2GRAY) / 255
    # epsilon = cv2.resize(epsilon, (192, 320))
    epsilon = cv2.resize(epsilon, (N, N))
    epsilon = torch.from_numpy(epsilon).to(device).unsqueeze(0).unsqueeze(0)

    epsilon = (epsilon.sub(0.5).sign() + 1) / 2 * 0.53 + 1
    epsilon = epsilon.to(torch.cfloat)
    epsilon.data.fill_(eps_si)
    epsilon.data[..., 10:30, 40:-40] = eps_si + 0.1
    epsilon.data[..., 70:90, 40:-40] = eps_si + 0.1

    grid_step = 0.05
    domain_size = (h * grid_step, w * grid_step)
    domain_size_pixel = [round(i / grid_step) for i in domain_size]
    print(domain_size_pixel)
    layer = FNO2d(
        in_channels=4,
        dim=16,
        kernel_list=[16, 16, 16, 16],
        kernel_size_list=[3, 3, 3, 3],
        padding_list=[1, 1, 1, 1],
        hidden_list=[128],
        # hidden_list=[256],
        # mode_list=[(48, 48)],
        mode_list=[(64, 64)],
        act_func="GELU",
        domain_size=domain_size,
        grid_step=grid_step,
        device=device,
    ).to(device)

    simulator = Simulation2D(
        mode="Hz",
        device_length=N,
        npml=4,
        use_dirichlet_bcs=True,
    )

    permittivities, src_x, src_y, Ex, Ey, Hz = simulator.solve(
        epsilon.real.cpu().numpy(), omega=OMEGA_1550, clip_buffers=True, src_x=1
    )

    E_target = Ex
    target_intensity = np.abs(E_target)
    phase = np.angle(E_target)
    target_img = np.concatenate([normalize(target_intensity), normalize(phase)], axis=1)
    target_intensity = torch.from_numpy(target_intensity).to(device)
    E_target = torch.from_numpy(E_target).to(device).to(torch.cfloat)

    # epsilon = torch.ones(1, 1, h, w, device=device, dtype=torch.cfloat)*1.5
    # epsilon -= 1j*0.00001

    light_source = torch.zeros(1, 1, h, w, device=device, dtype=torch.cfloat)
    MU0 = 4 * np.pi * 10 ** -7  # vacuum permeability
    light_source[..., light_source.size(-2) // 2, 1:2] = -1j * 1e6 * MU0 * wavelength
    obs = torch.cat([epsilon, light_source], dim=1)
    faraday_loss = FaradayResidue2d(grid_step=grid_step, wavelength=wavelength)
    mse_loss = IntensityMSELoss()
    optimizer = torch.optim.Adam(layer.parameters(), lr=3e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 3000, eta_min=1e-5, last_epoch=-1, verbose=False
    )

    for i in range(3000):
        E = layer(obs, wavelength)
        loss_pde = faraday_loss(E, epsilon, light_source)
        # loss_mse = mse_loss(E, target_intensity)
        loss_mse = torch.nn.functional.mse_loss(torch.view_as_real(E), torch.view_as_real(E_target))
        # loss = 0.005 * loss_pde + loss_mse
        loss = loss_mse
        # loss = loss_pde
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            print(
                f"step: {i}, loss = {loss.item()} loss_pde = {loss_pde.item()} loss_mse = {loss_mse.item()}"
            )
    E = E.data[0, 0]

    intensity = E.abs().cpu().numpy()
    phase = E.angle().cpu().numpy()
    # img = np.concatenate([normalize(intensity), normalize(phase)], axis=1)
    # plt.imshow(np.concatenate([target_img, img], axis=0), cmap="viridis")
    # plt.savefig("pde_mse_intensity_phase.png")

    fig, axes = plt.subplots(2, 3)
    E_target = E_target.data.cpu().numpy()

    axes[0, 0].imshow(normalize(np.abs(E_target)), cmap="viridis")
    axes[0, 0].title.set_text("FDFD Intensity")
    axes[1, 0].imshow(normalize(np.angle(E_target)), cmap="viridis")
    axes[1, 0].title.set_text("FDFD Phase")
    axes[0, 1].imshow(normalize(intensity), cmap="viridis")
    axes[0, 1].title.set_text("FNO Intensity")
    axes[1, 1].imshow(normalize(phase), cmap="viridis")
    axes[1, 1].title.set_text("FNO Phase")
    axes[0, 2].imshow(np.abs(np.abs(E_target) - intensity), cmap="viridis")
    axes[0, 2].title.set_text("Intensity Error")
    axes[1, 2].imshow(np.abs(np.angle(E_target) - phase), cmap="viridis")
    axes[1, 2].title.set_text("Phase Error")

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].xaxis.set_visible(False)
            axes[i, j].yaxis.set_visible(False)
    fig.suptitle(
        f"Hz Polarization. Ex field. {N*grid_step:.1f}um x {N*grid_step:.1f}um. eps=12.11. lambda=1550 nm"
    )
    plt.savefig("pde_mse_intensity_phase.png")


def learn_field_ref():
    import cv2

    device = torch.device("cuda:0")
    wavelength = 1.550

    img = cv2.imread("unitest/e_field_1.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.resize(img, (192, 320))
    N = 32
    img = cv2.resize(img, (N, N))
    h, w = img.shape[0], img.shape[1]
    img = torch.from_numpy(img).to(device).unsqueeze(0)
    h, w = img.shape[-2:]
    img = img.to(torch.cfloat)
    print(h, w)

    epsilon = cv2.imread("unitest/epsilon_field_1.png")
    epsilon = cv2.cvtColor(epsilon, cv2.COLOR_BGR2GRAY) / 255
    # epsilon = cv2.resize(epsilon, (192, 320))
    epsilon = cv2.resize(epsilon, (N, N))
    epsilon = torch.from_numpy(epsilon).to(device).unsqueeze(0).float()

    epsilon = (epsilon.sub(0.5).sign() + 1) / 2 * 0.53 + 1
    epsilon.data.fill_(eps_si)

    grid_step = 0.05
    domain_size = (h * grid_step, w * grid_step)
    domain_size_pixel = [round(i / grid_step) for i in domain_size]
    print(domain_size_pixel)

    layer = MaxwellSolver2D(
        size=N,
        src_x=1,
        src_y=N // 2,
    ).to(device)

    E = torch.nn.Parameter(torch.randn(1, h, w, device=device, dtype=torch.cfloat))
    E.data.copy_(img)
    E_target = E.data[0]
    target_intensity = E_target.abs().cpu().numpy()
    phase = E_target.angle().cpu().numpy()
    target_img = np.concatenate([normalize(target_intensity), normalize(phase)], axis=1)
    target_intensity = torch.from_numpy(target_intensity).to(device)

    optimizer = torch.optim.Adam(layer.parameters(), lr=2e-3)
    for i in range(2000):
        E = layer(epsilon)
        loss_pde = E.square().mean()
        loss = loss_pde
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"step: {i}, loss = {loss.item()} loss_pde = {loss_pde.item()}")

    E = layer.get_fields(epsilon).data[0].to(torch.cfloat)
    intensity = E.abs().cpu().numpy()
    phase = E.angle().cpu().numpy()
    img = np.concatenate([normalize(intensity), normalize(phase)], axis=1)
    plt.imshow(np.concatenate([target_img, img], axis=0), cmap="gray")
    plt.savefig("pde_intensity_phase_ref.png")


def learn_field_mmi():
    import cv2

    device = torch.device("cuda:0")
    wavelength, epsilon, field = torch.load("./data/mmi2x2/processed/training.pt")
    epsilon = epsilon.to(device)
    wavelength = wavelength.to(device).item()
    h, w = epsilon.shape[-2:]
    grid_step = 0.1
    domain_size = (h * grid_step, w * grid_step)
    domain_size_pixel = [round(i / grid_step) for i in domain_size]
    print(domain_size_pixel)
    layer = FNO2d(
        in_channels=8,
        out_channels=4,
        dim=16,
        kernel_list=[16, 16, 16, 16],
        kernel_size_list=[3, 3, 3, 3],
        padding_list=[1, 1, 1, 1],
        hidden_list=[128],
        # hidden_list=[256],
        # mode_list=[(48, 48)],
        mode_list=[(80, 80)],
        act_func="GELU",
        # act_func="SIREN",
        domain_size=domain_size,
        grid_step=grid_step,
        device=device,
    ).to(device)

    Ex = field[:, 0:1]  # [bs, 1, h, w]
    Ey = field[:, 1:2]
    Hz = field[:, -1:]

    E_target = Ex.to(device)
    Ey_target = Ey.to(device)
    Hz_target = Hz.to(device)
    target_intensity = E_target.abs()
    phase = E_target.angle()
    target_img = np.concatenate(
        [normalize(target_intensity).cpu().numpy(), normalize(phase).cpu().numpy()], axis=1
    )

    light_source = torch.zeros(1, 1, h, w, device=device, dtype=torch.cfloat)
    MU0 = 4 * np.pi * 10 ** -7  # vacuum permeability
    # light_source[..., 80:100, 1:20] = 1j * 1e6 * MU0 * wavelength
    # light_source[..., 100:102, 200:202] = 1j * 1e6 * MU0 * wavelength
    light_source = field[:, 0:3].to(device)
    # light_source = -1j*field[:, 0:1].to(device)
    # light_source[..., 100:] = 0
    obs = torch.cat([epsilon, light_source], dim=1)
    # obs = epsilon
    faraday_loss = FaradayResidue2d(grid_step=grid_step, wavelength=wavelength).to(device)
    tm_loss = TMPropagateFieldLoss(grid_step=grid_step, wavelength=wavelength).to(device)
    mse_loss = ComplexMSELoss()
    l1_loss = ComplexMSELoss()
    tv_loss = ComplexTVLoss()
    ps_loss = PhaseLoss()
    div_loss = DivergenceLoss()
    optimizer = torch.optim.Adam(layer.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 3000, eta_min=1e-5, last_epoch=-1, verbose=False
    )
    print(epsilon.shape)
    mag_Ex = E_target.abs().square().mean().item()
    mag_Ey = Ey_target.abs().square().mean().item()
    mag_Hz = Hz_target.abs().square().mean().item()
    print(mag_Ex, mag_Ey, mag_Hz)
    print(mag_Ex / mag_Hz, mag_Ey / mag_Hz)
    Ex, Ey = tm_loss.Hy_to_E(Hz_target, epsilon)
    print(Ex.abs().mean().item())
    print(Ey.abs().mean().item())
    exit(0)
    for i in range(2000):
        E = layer(obs, wavelength)
        # print(E.device, epsilon.device, light_source.device)
        # loss_pde = faraday_loss(E[:, 0:1], epsilon, light_source)
        loss_pde = torch.zeros(1, device=E.device)
        # res_x = faraday_loss(E[:, 0:1], epsilon, source_field=None, reduction="none")
        # res_y = faraday_loss(Hz_target, epsilon, source_field=None, reduction="none")
        # loss_res = l1_loss(res_x, res_y)
        loss_div = div_loss(E[:, 0:1], epsilon)
        res_x = tm_loss(E[:, 0:1], epsilon, source_field=None, reduction="none")
        res_y = tm_loss(Hz_target, epsilon, source_field=None, reduction="none")
        loss_res = l1_loss(res_x, res_y)
        # print(E[:, 0:1].abs().mean().item())
        # print(Hz_target.abs().mean().item())
        # print(loss_tm_gt.item())
        # print(loss_tm.item())
        # exit(0)
        # print(loss_tm.item())
        # exit(0)
        # loss_mse = mse_loss(E, target_intensity)

        loss_l1 = l1_loss(E[:, 0:1], Hz_target)
        # loss_tv = tv_loss(E[:, 0:1], E_target) + tv_loss(E[:, 1:2], Ey_target)
        loss_tv = torch.zeros(1, device=E.device)
        # loss_ps = ps_loss(E[:, 0:1], E_target)# + ps_loss(E[:, 1:2], Ey_target)
        loss_ps = torch.zeros(1, device=E.device)
        # loss = 0.00005 * loss_pde + loss_l1
        # loss = loss_l1 + 0.1 * loss_tv + 0.1 * loss_ps
        # loss = 0.001*loss_res + loss_l1 +
        loss = 0.00 * loss_res + loss_l1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            print(
                f"step: {i}, loss = {loss.item()} loss_tv = {loss_tv.item()} loss_l1 = {loss_l1.item()} loss_ps = {loss_ps.item()} loss_pde = {loss_pde.item()} loss_res = {loss_res.item()}"
            )
    # light_source = field[:, 0:3].to(device)
    # light_source[..., 100:] = 0
    # light_source = torch.flip(light_source, [2])
    # obs = torch.cat([epsilon, light_source], dim=1)
    # E = layer(obs, wavelength)
    E = E.data[0, 0]

    intensity = E.abs().cpu().numpy()
    phase = E.angle().cpu().numpy()
    # img = np.concatenate([normalize(intensity), normalize(phase)], axis=1)
    # plt.imshow(np.concatenate([target_img, img], axis=0), cmap="viridis")
    # plt.savefig("pde_mse_intensity_phase.png")

    fig, axes = plt.subplots(2, 3)
    E_target = Hz_target[0, 0].data.cpu().numpy()

    axes[0, 0].imshow(normalize(np.abs(E_target)), cmap="viridis")
    axes[0, 0].title.set_text("FDFD Intensity")
    axes[1, 0].imshow(normalize(np.angle(E_target)), cmap="viridis")
    axes[1, 0].title.set_text("FDFD Phase")
    axes[0, 1].imshow(normalize(intensity), cmap="viridis")
    axes[0, 1].title.set_text("FNO Intensity")
    axes[1, 1].imshow(normalize(phase), cmap="viridis")
    axes[1, 1].title.set_text("FNO Phase")
    axes[0, 2].imshow(np.abs(np.abs(E_target) - intensity), cmap="viridis")
    axes[0, 2].title.set_text("Intensity Error")
    axes[1, 2].imshow(np.abs(np.angle(E_target) - phase), cmap="viridis")
    axes[1, 2].title.set_text("Phase Error")

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].xaxis.set_visible(False)
            axes[i, j].yaxis.set_visible(False)
    fig.suptitle(
        f"Hz Polarization. Ex field. {h*grid_step:.1f}um x {w*grid_step:.1f}um. eps=12.11. lambda=1550 nm"
    )
    # plt.show()
    plt.savefig("mmi2x2_pde_mse_intensity_phase.png")


def learn_simu_mmi():
    wavelength, data, target = torch.load("./data/mmi/processed/training.pt")
    device = torch.device("cuda:0")
    lambda0 = 1.55
    n = 6
    wavelength = wavelength[:n].to(device)
    print(wavelength)
    data = data[:n,].to(device)
    # data = data[-n:, :, 10:-10, :-100].to(device)
    target = target[:n].to(device)
    # target = target[-n:, :, 10:-10, :-100].to(device)
    grid_step = 0.1
    domain_size = [8, 52]
    # domain_size = [6, 42]
    n_fields = 3
    layer = FNO2d(
        in_channels=8,
        out_channels=n_fields*2,
        dim=16,
        kernel_list=[16, 16, 16, 16],
        kernel_size_list=[3, 3, 3, 3],
        padding_list=[1, 1, 1, 1],
        hidden_list=[128],
        # hidden_list=[256],
        # mode_list=[(48, 48)],
        mode_list=[(80,200)],
        # mode_list=[(50,160)],
        act_func="GELU",
        # act_func="SIREN",
        domain_size=domain_size,
        grid_step=grid_step,
        buffer_width=0,
        device=device,
    ).to(device)
    print(sum(i.numel() for i in layer.parameters()))
    # layer = EncoderDecoder(
    #     in_channels=4,
    #     out_channels=2,
    #     dim=16,
    #     kernel_list=[16, 16, 16, 16],
    #     kernel_size_list=[3, 3, 3, 3],
    #     padding_list=[1, 1, 1, 1],
    #     hidden_list=[128],
    #     act_func="GELU",
    #     # act_func="SIREN",
    #     domain_size=domain_size,
    #     grid_step=grid_step,
    #     buffer_width=0,
    #     dropout_rate=0.2,
    #     device=device,
    # ).to(device)
    l1_loss = ComplexMSELoss()
    tm_loss = TMPropagateFieldLoss(grid_step=grid_step, wavelength=wavelength[0,0].item()).to(device)
    fr_loss = FaradayResidue2d(grid_step=grid_step, wavelength=wavelength[0,0].item())
    tv_loss = ComplexTVLoss()
    curl_loss = CurlLoss(grid_step=grid_step, wavelength=wavelength[0,0].item())
    print(curl_loss(target[0:1], data[0:1]))
    # exit(0)
    # target = torch.stack([target[:,0],-target[:,1],target[:,2]], dim=1)
    # print(curl_loss(target[0:1]))
    # print(target[0:1, 1:2].abs().square().sum(dim=2))
    # exit(0)
    optimizer = torch.optim.Adam(layer.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 3000, eta_min=1e-5, last_epoch=-1, verbose=False
    )
    print(target[:, -1:].abs().min(), target[:, -1:].abs().max())
    # obs = torch.cat([data[:, 0:1].sqrt(), data[:, -1:]], dim=1)
    obs = data
    # obs = data[:, 0:1].sqrt()
    for i in range(3500):
        output = layer(obs, wavelength)
        loss_l1 = l1_loss(output[:, -n_fields:], target[:, -n_fields:])
        # loss_tm = l1_loss(tm_loss(output[:, -n_fields:], data[:, 0:1], reduction="none"), tm_loss(target[:, -n_fields:], data[:, 0:1], reduction="none"))
        loss_tm = torch.zeros(1, device=device)
        # loss_fr = l1_loss(fr_loss(output[:, -n_fields:], data[:, 0:1], reduction="none"), fr_loss(target[:, -n_fields:], data[:, 0:1], reduction="none"))
        loss_fr = torch.zeros(1, device=device)
        # loss_tv = tv_loss(output[:, -n_fields:], target[:, -n_fields:])
        loss_tv = torch.zeros(1, device=device)
        loss_curl = l1_loss(curl_loss(output, data, reduction="none"), curl_loss(target, data, reduction="none"))
        loss_en = output.abs().square().sum(dim=2).std(dim=-1).mean()
        loss = loss_l1 + 1e-5 * loss_curl
        # if i < 3000:
        #     loss = loss_l1
        # else:
        #     loss = 1e-5*loss_curl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            print(f"step: {i}, loss = {loss.item():.6f} loss_l1 = {loss_l1.item():.6f} loss_tm = {loss_tm.item():.6f} loss_fr = {loss_fr.item():.6f} loss_tv = {loss_tv.item():.6f} loss_curl = {loss_curl.item():.6f} loss_en = {loss_en.item():.6f}")

    fig, axes = plt.subplots(4, min(n,4), figsize=(15,15))
    for i in range(axes.shape[1]):
        ch = 1
        E_target = target[i, ch].data.cpu().numpy()
        E = output.data[i, ch]
        intensity = E.abs().cpu().numpy()
        # phase = E.angle().cpu().numpy()

        axes[0, i].imshow(normalize(np.abs(E_target)), cmap="viridis")
        axes[0, i].title.set_text("FDFD Intensity")
        axes[1, i].imshow(normalize(intensity), cmap="viridis")
        axes[1, i].title.set_text("FNO Intensity")
        axes[2, i].imshow(np.abs(np.abs(E_target) - intensity), cmap="viridis")
        axes[2, i].title.set_text("Intensity Error")
        axes[3, i].imshow(data[0,0].abs().cpu().numpy(), cmap="viridis")

        for j in range(axes.shape[0]):
            axes[j, i].xaxis.set_visible(False)
            axes[j, i].yaxis.set_visible(False)
    fig.suptitle(
        f"TM: Hz Pol. ({['Ex', 'Ey', 'Hz'][ch]}). eps=12.11. lambda=1550 nm"
    )
    # plt.show()
    plt.tight_layout()
    plt.savefig("mmi_learn_simu.png", dpi=300)


if __name__ == "__main__":
    # learn_field()
    # learn_field_mmi()
    # learn_field_ref()
    learn_simu_mmi()
