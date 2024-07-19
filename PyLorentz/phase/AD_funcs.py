import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from PyLorentz.visualize import show_im
import torchvision.transforms as T
from .AD_utils import Tukey2D


def training_loop(
    inp_im,
    inp_amp,
    inp_phase,
    TFs,
    pdict,
    dipA=None, # opportunity to define custom DIPs
    dipP=None, # default None + pdict['DIP'] == True -> defaults implemented
    blurrer=None, # primarily for Amp -- true will make one from pdict['blur_sigma']
    print_every=50,
):
    now = datetime.datetime.now().strftime('%y%m%d-%H%M')
    if pdict['DIP']:
        pdict['runtype'] = "DIPAD" # vs CNN, for later
        inp_phase.requires_grad = False
        inp_phase = inp_phase.unsqueeze(0)
        LRs = pdict["LRs"][2]
    else:
        pdict['runtype'] = "AD" # vs CNN, for later
        inp_phase.requires_grad = True
        if dipP is not None:
            print("dipP was given but pdict['DIP'] == False, setting dipP to None")
            dipP = None
        if pdict['solve_amp'] and dipA is not None:
            print("dipA was given but pdict['DIP'] == False, setting dipA to None")
            dipA = None
        LRs = pdict["LRs"][0]

    if pdict['solve_amp']:
        pdict['runtype'] += "-amp"
        if pdict['DIP']:
            inp_amp.requires_grad = False
            inp_amp = inp_amp.unsqueeze(0)
        else:
            inp_amp.requires_grad = True

    if pdict['mode'] == "sim":
        pdict['name'] = f"{now}_{pdict['dataname']}_df{','.join(map(str,pdict['defvals']/1e6))}mm_PN{pdict['noise_vals']['gauss']}_{pdict['runtype']}"
    else:
        pdict['name'] = f"{now}_{pdict['dataname']}_df{','.join(map(str,pdict['defvals']/1e6))}mm_{pdict['runtype']}"

    print(f"Beginning run for {pdict['name']}")
    print(f"Defocus values (mm): {pdict['defvals']/1e6}")
    print(pdict['notes'])

    if "window" not in pdict.keys():
        pdict['window'] = False

    if print_every == 0:
        print_every = -1

    # TODO make sure all LR weights are >= 0

    if pdict['solve_amp']:
        # run training loop amp
        pdict['amp_mode'] = pdict['amp_mode'].lower()
        if pdict['amp_mode'] not in ["binary", "float", "integer"]:
            raise ValueError(f"pdict['amp_mode'] bad value of {pdict['amp_mode']}")

        return _training_loop_amp(
            inp_im,
            inp_amp,
            inp_phase,
            TFs,
            pdict,
            dipP,
            dipA,
            blurrer,
            print_every,
            )



    else:
        return _training_loop_noamp(
            inp_im,
            inp_amp,
            inp_phase,
            TFs,
            pdict,
            dipP,
            blurrer,
            print_every,
            )

    return

def get_scheduler(optimizer, pdict):
    pdict['scheduler'] = pdict['scheduler'].lower()
    if pdict['DIP']:
        LRs = pdict['LRs'][2]
    else:
        LRs = pdict['LRs'][0]

    if pdict['scheduler'] == "none":
        scheduler = None
    elif pdict['scheduler'] == "cyclic":
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                        base_lr=LRs[0]/2,
                                        max_lr=LRs[0]*5,
                                        step_size_up=100,
                                        mode="triangular2",
                                        cycle_momentum=False)
    elif pdict['scheduler'] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode='min',
                                                            factor=0.75,
                                                            patience=pdict['plateau_patience'],
                                                            threshold=1e-4,
                                                            min_lr = LRs[0]/20,
                                                            verbose=True)
    elif pdict['scheduler'] == "exp":
        gamma = 0.9997
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    return scheduler

def _training_loop_noamp(
    inp_im,
    inp_amp,
    inp_phase,
    TFs,
    pdict,
    dip,
    blurrer=None,
    print_every=1,
):

    maxiter = pdict['maxiter']
    (optLR, tvLR, dipLR) = pdict['LRs']

    if pdict['DIP']:
        optimizer = torch.optim.Adam([
            {'params': dip.parameters(), 'lr':dipLR[0]},
            ])
        scheduler = get_scheduler(optimizer, pdict)
    else:
        guess_phase = inp_phase
        guess_phase.requires_grad = True
        optimizer = torch.optim.Adam([
            {'params': guess_phase, 'lr':optLR[0]}
            ])
        scheduler = get_scheduler(optimizer, pdict)

    inp_amp.requires_grad = False
    guess_amp = inp_amp
    if guess_amp.max() != guess_amp.min():
        amp_phase = amp_to_phi(guess_amp, pdict['AMP2PHI'])
    else:
        amp_phase = torch.zeros_like(guess_amp, requires_grad=False)

    if pdict['window']:
        window = torch.tensor(
            Tukey2D(inp_im[0].cpu().detach().numpy().shape),
            device=pdict['device'], dtype=torch.float32)
    else:
        window = torch.ones_like(inp_im)

    losses = []
    minloss = 9e9
    stime = time.time()
    ttime = time.time()
    for i in range(maxiter):
        if pdict['DIP']:
            if i == 0:
                if pdict['pre-train_dip'] > 0:
                    print(f"pre-training dip for {pdict['pre-train_dip']} iters")
                    for _ in range(pdict['pre-train_dip']):
                        # if pdict['DIP_noise_frac'] > 0:
                        #     inp_phase += pdict['DIP_noise_frac'] * torch.randn(inp_phase.shape,
                        #                                                 device=inp_phase.device)
                        loss = train_DIP(inp_phase, torch.ones_like(inp_phase), dip)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
            if pdict['DIP_noise_frac'] > 0:
                inp_phase += pdict['DIP_noise_frac'] * torch.randn(inp_phase.shape,
                                                            device=inp_phase.device)
            if blurrer is None or not pdict['use_blur']:
                guess_phase = dip.forward(inp_phase)[0]
            else:
                guess_phase = blurrer(dip.forward(inp_phase))[0]

        loss = compute_loss(inp_im*window, guess_amp*window, guess_phase*window, TFs, tvLR, amp_phase, pdict)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append([loss.item(), optimizer.param_groups[0]["lr"]])

        if (i == 0 or (i + 1) % print_every == 0) and print_every > 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"""{i + 1}/{maxiter}, {time.time() - ttime:.2f} s : loss {loss.item():.3e}, lr {lr:.2e}"""
            )
            ttime = time.time()

            if pdict['show']:
                show_im((guess_phase-guess_phase.min()).cpu().detach().numpy(), f"phase, iter {i+1}")
                show_im((sim_images_torch(guess_phase - amp_phase, TFs, guess_amp)).cpu().detach().numpy(), f"guess image, iter {i+1}")


        if scheduler is not None:
            if hasattr(scheduler, "cooldown"): # is plateau
                scheduler.step(loss.item())
            else:
                scheduler.step()

        if i > 100 and loss.item() < minloss:
            best_phase = guess_phase.detach().clone()
            best_amp = guess_amp.detach().clone()
            minloss = loss.item()
    print(
        f"""total time (h:m:s) = {datetime.timedelta(seconds=round(time.time() - stime))}"""
    )
    guess_phase = guess_phase - torch.min(guess_phase)
    best_phase = best_phase - torch.min(best_phase)
    return (guess_amp, guess_phase, best_amp, best_phase, np.array(losses))


def _training_loop_amp(
    inp_im,
    inp_amp,
    inp_phase,
    TFs,
    pdict,
    dipP=None,
    dipA=None,
    blurrer=None,
    print_every=1,
):
    maxiter = pdict['maxiter']
    (optLR, tvLR, dipLR) = pdict['LRs']

    if pdict['DIP']:
        optimizer = torch.optim.Adam([
                    {'params': dipP.parameters(), 'lr':dipLR[0]},
                    {'params': dipA.parameters(), 'lr':dipLR[1]}
                    ])
        scheduler = get_scheduler(optimizer, pdict)
    else:
        guess_phase = inp_phase
        # guess_phase.requires_grad = True
        guess_amp = inp_amp
        # guess_amp.requires_grad = True
        optimizer = torch.optim.Adam([
            {'params': guess_phase, 'lr':optLR[0]},
            {'params': guess_amp, 'lr':optLR[1]},
            # {'params': pdict['AMP2PHI'], 'lr':0.0002}
            ])
        scheduler = get_scheduler(optimizer, pdict)

    if pdict['window']:
        window = torch.tensor(
            Tukey2D(inp_im[0].cpu().detach().numpy().shape, alpha=0.1),
            device=pdict['device'], dtype=torch.float32)
        if pdict['show']:
            show_im(window.cpu().detach().numpy(), "window")
    else:
        window = torch.ones_like(inp_im[0])


    losses = []
    minloss = 9e9
    stime = time.time()
    ttime = time.time()
    for i in range(maxiter):
        if pdict['DIP']:
            if i == 0:
                if pdict['pre-train_dip'] > 0:
                    print(f"pre-training dipP and dipA for {pdict['pre-train_dip']} iters")
                    for _ in range(pdict['pre-train_dip']):
                        lossP = train_DIP(inp_phase, torch.ones_like(inp_amp), dipP)
                        lossA = train_DIP(inp_amp, torch.ones_like(inp_amp), dipA)
                        loss = lossP + lossA
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    if pdict['show']:
                        show_im(dipP.forward(inp_phase)[0].cpu().detach().numpy(), 'guess phase after pretrain')
                        show_im(dipA.forward(inp_amp)[0].cpu().detach().numpy(), 'guess amp after pretrain')

                    optimizer = torch.optim.Adam([
                        {'params': dipP.parameters(), 'lr':dipLR[0]},
                        ])
                    scheduler = get_scheduler(optimizer, pdict)

            if i+1 == pdict['start_amp']:
                print('starting amplitude learning')
                optimizer = torch.optim.Adam([
                    {'params': dipP.parameters(), 'lr':dipLR[0]},
                    {'params': dipA.parameters(), 'lr':dipLR[1]},
                    # {'params': pdict['AMP2PHI'], 'lr':0.002}
                    ])
                scheduler = get_scheduler(optimizer, pdict)

            if pdict['DIP_noise_frac'] > 0:
                inp_phase += pdict['DIP_noise_frac'] * torch.randn(inp_phase.shape,
                                                            device=inp_phase.device)
                if i >= pdict['start_amp']:
                    inp_amp += pdict['DIP_noise_frac'] * torch.randn(inp_amp.shape,
                                                                device=inp_amp.device)
            guess_amp = dipA.forward(inp_amp)[0]
            guess_phase = dipP.forward(inp_phase)[0]

        guess_amp = torch.abs(guess_amp) # enforce that amp > 0

        if pdict['amp_mode'] == "float":
            pass
        elif pdict['amp_mode'] == "binary":
            if (i+1) >= pdict['start_binary'] and (i+1)%pdict['binary_every'] == 0 and pdict['start_binary']>0:
                if blurrer is not None and pdict['use_blur']:
                    guess_amp = binary_amp(blurrer(guess_amp.unsqueeze(0))[0])
                else:
                    guess_amp = binary_amp(guess_amp)
        elif pdict['amp_mode'] == "integer":
            raise NotImplementedError


        amp_phase = amp_to_phi(guess_amp, pdict['AMP2PHI'])

        if blurrer is not None and pdict['use_blur']:
            guess_amp = blurrer(guess_amp.unsqueeze(0))[0]
            guess_phase = blurrer(guess_phase.unsqueeze(0))[0]
            amp_phase = blurrer(amp_phase.unsqueeze(0))[0]

        # guess_amp = torch.abs(guess_amp) # enforce that amp > 0

        guess_phase = torch.abs(guess_phase)
        guess_phase *= window
        # guess_amp *= window

        loss = compute_loss(inp_im*window, guess_amp, guess_phase, TFs, tvLR, amp_phase, pdict)
        if not loss.item()>0:
            print(f" iter {i} loss.item: {loss.item()}")
            if torch.isnan(loss.item()):
                print(f"nan from loss, iter {i}")
                raise ValueError

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append([loss.item(), optimizer.param_groups[0]["lr"]])

        if (i == 0 or (i + 1) % print_every == 0) and print_every > 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"""{i + 1}/{maxiter}, {time.time() - ttime:.2f} s : loss {loss.item():.3e}, lr {lr:.6f}"""
            )
            print("AMP2PHI: ", pdict['AMP2PHI'])
            ttime = time.time()

            if pdict['show']:
                show_im(guess_amp.cpu().detach().numpy(), f"amp, iter {i+1}")
                show_im((guess_phase-guess_phase.min()).cpu().detach().numpy(), f"phase, iter {i+1}")

        if scheduler is not None:
            if hasattr(scheduler, "cooldown"): # is plateau
                scheduler.step(loss.item())
            else:
                scheduler.step()

        if i > 100 and loss.item() < minloss:
            best_phase = guess_phase.detach().clone()
            best_amp = guess_amp.detach().clone()
            minloss = loss.item()
    print(
        f"""total time (h:m:s) = {datetime.timedelta(seconds=round(time.time() - stime))}"""
    )
    guess_phase = guess_phase - torch.min(guess_phase)
    best_phase = best_phase - torch.min(best_phase)
    return (guess_amp, guess_phase, best_amp, best_phase, np.array(losses))



def compute_loss(inp_im, guess_amp, guess_phase, TF, TVLR, amp_phase, pdict):
    # negative phase shift because of how i'm simulating input images
    # sign doesn't really matter, for input images looks better this way
    guess_im = sim_images_torch(guess_phase - amp_phase, TF, guess_amp)
    MSE_loss = torch.mean((guess_im - inp_im) ** 2)
    if pdict['DIP'] or (TVLR[0] == 0 and TVLR[1] == 0):
        return MSE_loss
    TV_loss = calc_TV_loss_PBC(guess_amp, guess_phase, TVLR, pdict['solve_amp'])
    return MSE_loss + TV_loss


def train_DIP(inp_phase, mask, dip):
    masked_phase = inp_phase * mask
    dip_out = dip.forward(masked_phase)[0]
    masked_output = mask * dip_out
    return torch.mean((masked_output - masked_phase)**2)

def compute_loss_retboth(inp_im, guess_amp, inp_phase, TF, TVLR, solve_amp):
    """
    returns MSE loss and TV loss seperately so they can be tracked each
    """
    guess_im = sim_images_torch(inp_phase, TF, guess_amp)
    MSE_loss = torch.mean((guess_im - inp_im) ** 2)
    if TVLR[0] == 0 and TVLR[1] == 0:
        TV_loss = torch.tensor(0)
    else:
        TV_loss = calc_TV_loss_PBC(guess_amp, inp_phase, TVLR, solve_amp)
    return guess_im, MSE_loss, TV_loss


def sim_images_torch(tphis, TFs, Amps):
    """Simulate images for a phase tensor. Z-size is number of images in the stack.

    Args:
        tphis (Tensor): Stack of phase shifts with dimension [tsize, dy, dx]
        TFs (Tensor): Stack of microscope transfer functions, [tsize, dy, dx]
        Amps (Tensor): Stack of amplitude functions [tsize, dy, dx]

    Returns:
        Tensor: Stack of images [tsize, dy, dx]
    """
    ObjWave = Amps * torch.exp(tphis * 1j)  # keep * 1j at end in case want to jit
    ImgWaves = torch.fft.ifft2(torch.fft.fft2(ObjWave) * TFs)
    Imgs = torch.abs(ImgWaves) ** 2
    return Imgs


def calc_TV_loss(guess_amp, inp_phase, weights, solve_amp):
    if inp_phase.ndim == 2:
        (dy, dx) = inp_phase.size()
        tv_phase = None
        if weights[0] != 0:  # calc phase TV
            tv_phase_h = torch.pow(inp_phase[1:, :] - inp_phase[:-1, :], 2).sum()
            tv_phase_w = torch.pow(inp_phase[:, 1:] - inp_phase[:, :-1], 2).sum()
            tv_phase = weights[0] * (tv_phase_h + tv_phase_w) / (dy * dx)

        if solve_amp and weights[1] != 0:  # amp TV
            tv_amp_h = torch.pow(guess_amp[1:, :] - guess_amp[:-1, :], 2).sum()
            tv_amp_w = torch.pow(guess_amp[:, 1:] - guess_amp[:, :-1], 2).sum()
            tv_amp = weights[1] * (tv_amp_h + tv_amp_w) / (dy * dx)
            if tv_phase is None:
                return tv_amp
            else:
                return (tv_amp + tv_phase) / 2
        else:
            return tv_phase

    raise NotImplementedError("Havne't implemented TV loss for higher dims yet")


def calc_TV_loss_PBC(guess_amp, inp_phase, weights, solve_amp):
    """
    Modified to include PBCs in the TV loss
    """
    if inp_phase.ndim in [2,3]:
        if inp_phase.ndim == 3:
            if inp_phase.shape[0] == 1:
                inp_phase = inp_phase.squeeze()
            else:
                raise NotImplementedError
        (dy, dx) = inp_phase.size()
        tv_phase = None
        if weights[0] != 0:  # calc phase TV
            phase_pad_h = F.pad(
                inp_phase[None, None, ...], (0, 0, 0, 1), mode="circular"
            )[0, 0]
            phase_pad_w = F.pad(
                inp_phase[None, None, ...], (0, 1, 0, 0), mode="circular"
            )[0, 0]

            tv_phase_h = torch.pow(phase_pad_h[1:, :] - phase_pad_h[:-1, :], 2).sum()
            tv_phase_w = torch.pow(phase_pad_w[:, 1:] - phase_pad_w[:, :-1], 2).sum()
            tv_phase = weights[0] * (tv_phase_h + tv_phase_w) / (dy * dx)

        if solve_amp and weights[1] != 0:  # amp TV
            amp_pad_h = F.pad(
                guess_amp[None, None, ...], (0, 0, 0, 1), mode="circular"
            )[0, 0]
            amp_pad_w = F.pad(
                guess_amp[None, None, ...], (0, 1, 0, 0), mode="circular"
            )[0, 0]

            tv_amp_h = torch.pow(amp_pad_h[1:, :] - amp_pad_h[:-1, :], 2).sum()
            tv_amp_w = torch.pow(amp_pad_w[:, 1:] - amp_pad_w[:, :-1], 2).sum()
            tv_amp = weights[1] * (tv_amp_h + tv_amp_w) / (dy * dx)
            if tv_phase is None:
                return tv_amp
            else:
                return (tv_amp + tv_phase) / 2
        else:
            return tv_phase

    raise NotImplementedError("Havne't implemented TV loss for higher dims yet")

def random_mask(shape, device, frac=0.5):
    mask = torch.rand(shape, device=device)
    mask[mask<frac] = 0
    mask[mask!=0] = 1
    return mask


def amp_to_phi_np(amp, AMP2PHI):
    a = -1*np.log(amp)
    a = a - np.min(a)
    return a * AMP2PHI

def amp_to_phi(amp, AMP2PHI):
    a = -1*torch.log(amp)
    b = a - torch.min(a)
    return b * AMP2PHI


def binary_amp(amp, threshval = None):
    ampr = amp.ravel()
    highval = torch.mode(ampr)[0]
    if threshval is None:
        threshval = 3*highval / 4 #  1/3
    s, _ = torch.sort(ampr)
    lowval = torch.min(ampr)

    # sind = torch.argwhere(s > threshval)
    # lowval = torch.mean(s[:sind[0]])

    return torch.where(amp > threshval, highval, lowval)
    # h = (amp > threshval)*highval
    # l = (amp <= threshval)*lowval
    # return h + l
