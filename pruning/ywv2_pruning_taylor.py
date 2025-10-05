import argparse
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Union
import numpy as np
import torch, re, ast
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics import YOLO, YOLOWorld, __version__
from ultralytics.nn.modules import WorldDetect, C2f, Conv, Bottleneck, C2fAttn, DFL, SEBlock, BNContrastiveHead, Detect, Concat, PhiNet, PhiNetConvBlock
from ultralytics.nn.modules.block import MaxSigmoidAttnBlock
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils.torch_utils import initialize_weights, de_parallel
from ultralytics.utils import LOGGER, yaml_load, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, RANK
from ultralytics.utils.checks import check_yaml
from torchvision.models import resnet18
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.world.train import WorldTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
import torch_pruning as tp



def _find_projector_in(attn, c_attn, gl_in):
    """
    Trova un layer tra proj_conv e gl che abbia IN == c_attn e OUT == gl_in.
    Ritorna il modulo (Conv2d o Linear) oppure None.
    """
    for m in attn.modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == c_attn and m.out_channels == gl_in:
            return m
        if isinstance(m, nn.Linear) and getattr(m, 'in_features', None) == c_attn and getattr(m, 'out_features', None) == gl_in:
            return m
    return None

def _rebuild_linear_in(linear: nn.Linear, keep_in_idx: torch.Tensor) -> nn.Linear:
    new = nn.Linear(in_features=len(keep_in_idx), out_features=linear.out_features, bias=linear.bias is not None)
    with torch.no_grad():
        new.weight.copy_(linear.weight[:, keep_in_idx])
        if linear.bias is not None:
            new.bias.copy_(linear.bias)
    return new

def fix_proj_bn_after_head_prune(model):
    from ultralytics.nn.modules.block import MaxSigmoidAttnBlock
    for m in model.modules():
        if isinstance(m, MaxSigmoidAttnBlock):
            oc = m.proj_conv.conv.out_channels
            bn = m.proj_conv.bn
            if bn.num_features != oc:
                new_bn = nn.BatchNorm2d(
                    oc, eps=bn.eps, momentum=bn.momentum, affine=True, track_running_stats=True
                )
                with torch.no_grad():
                    k = min(oc, bn.weight.numel())
                    new_bn.weight[:k].copy_(bn.weight[:k])
                    new_bn.bias[:k].copy_(bn.bias[:k])
                    new_bn.running_mean[:k].copy_(bn.running_mean[:k])
                    new_bn.running_var[:k].copy_(bn.running_var[:k])
                    # Se oc > k, i restanti canali restano ai default (γ=1, β=0, mean=0, var=1)
                m.proj_conv.bn = new_bn

def fix_attn_invariants(model):
    for m in model.modules():
        if isinstance(m, MaxSigmoidAttnBlock):
            nh = m.nh
            c2 = m.proj_conv.conv.out_channels  # canali attesi dall'attn output

            # Assicura che ec produca esattamente c2 canali
            if m.ec is None:
                c1 = m.proj_conv.conv.in_channels
                if c1 != c2:
                    # crea un 1x1 per mappare c1 -> c2
                    new_ec = Conv(c1, c2, k=1, act=False)
                    m.ec = new_ec
            else:
                old_ec = m.ec
                ec_out = old_ec.conv.out_channels
                if ec_out != c2:
                    new_ec = Conv(old_ec.conv.in_channels, c2, k=1, act=False)
                    # copia parziale dei pesi (shrinking o padding zero se serve)
                    with torch.no_grad():
                        oc = min(ec_out, c2)
                        new_ec.conv.weight[:oc] = old_ec.conv.weight[:oc]
                        for attr in ['weight','bias','running_mean','running_var']:
                            getattr(new_ec.bn, attr)[:oc] = getattr(old_ec.bn, attr)[:oc]
                        if c2 > ec_out:
                            # canali aggiuntivi a zero
                            new_ec.conv.weight[oc:] = 0
                            getattr(new_ec.bn, 'weight')[oc:] = 1
                            getattr(new_ec.bn, 'bias')[oc:] = 0
                            getattr(new_ec.bn, 'running_mean')[oc:] = 0
                            getattr(new_ec.bn, 'running_var')[oc:] = 1
                    m.ec = new_ec

            # Allinea la linear 'gl' per generare ec == c2
            if m.gl.out_features != c2:
                old_gl = m.gl
                new_gl = nn.Linear(old_gl.in_features, c2, bias=True)
                with torch.no_grad():
                    of = min(old_gl.out_features, c2)
                    new_gl.weight[:of] = old_gl.weight[:of]
                    new_gl.bias[:of]   = old_gl.bias[:of]
                    if c2 > of:
                        new_gl.weight[of:] = 0
                        new_gl.bias[of:]   = 0
                m.gl = new_gl

            # Recompute head channels e controlli
            assert c2 % nh == 0, f"proj_conv.out_channels ({c2}) non divisibile per nh ({nh})"
            m.hc = c2 // nh

def _parse_zeroed_heads(log_path: Path):
    """
    Ritorna { c2fattn_module_name: sorted(list(head_ids_to_drop)) }
    Legge SOLO righe tipo: "<name>.attn.proj_conv.conv|out|[idx,...]"
    """
    import re, ast
    pat = re.compile(r'^(.*)\.attn\.proj_conv\.conv\|out\|\[(.*)\]$')
    heads_to_drop = {}
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            m = pat.match(line)
            if not m:
                continue
            base, idx_str = m.groups()
            # base è il nome del modulo C2fAttn (es: "model.14")
            # idx sono i canali out prunati nel proj_conv (sono gruppi da hc)
            idx_list = ast.literal_eval(f'[{idx_str}]') if idx_str else []
            if not idx_list:
                continue
            # salviamo le channel idx; il mapping a head lo facciamo dopo vedendo hc
            heads_to_drop.setdefault(base, []).extend(idx_list)
    # rimuovi duplicati
    for k in list(heads_to_drop.keys()):
        heads_to_drop[k] = sorted(set(heads_to_drop[k]))
    return heads_to_drop

def _get_module_by_name(model: nn.Module, dotted: str):
    """Ritorna (module, parent_module, attr_name) dato un path tipo 'model.14'."""
    parts = dotted.split(".")
    cur = model
    parent = None
    attr_name = None
    for p in parts:
        parent = cur
        if p.isdigit():
            cur = cur[int(p)]
            attr_name = p
        else:
            cur = getattr(cur, p)
            attr_name = p
    return cur, parent, attr_name


def _rebuild_conv_out(conv: nn.Conv2d, keep_out_idx: torch.Tensor) -> nn.Conv2d:
    """Ricostruisce Conv2d con meno out_channels (selezionando righe weight)."""
    new = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=len(keep_out_idx),
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        new.weight.copy_(conv.weight[keep_out_idx, ...])
        if conv.bias is not None:
            new.bias.copy_(conv.bias[keep_out_idx])
    return new


def _rebuild_conv_in(conv: nn.Conv2d, keep_in_idx: torch.Tensor) -> nn.Conv2d:
    """Ricostruisce Conv2d con meno in_channels (selezionando colonne weight)."""
    new = nn.Conv2d(
        in_channels=len(keep_in_idx),
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1 if conv.groups == conv.in_channels else conv.groups,  # niente depthwise qui
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        new.weight.copy_(conv.weight[:, keep_in_idx, ...])
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def structurally_prune_c2fattn_heads(model: nn.Module, log_path: Path):
    """
    Taglia DAVVERO le head azzerate nei blocchi C2fAttn:
      - riduce out_channels di attn.proj_conv.conv
      - riduce gli in_channels di cv2 (sulla fetta 'attn' della concat)
      - aggiorna nh/bias/scale in MaxSigmoidAttnBlock
    Si basa su zeroed_params.txt prodotto in fase di zeroing.
    """
    # 1) raccogli canali out zeroed per ogni blocco
    base_to_zeroed_ch = _parse_zeroed_heads(log_path)
    if not base_to_zeroed_ch:
        LOGGER.info("No attention heads to structurally prune (no proj_conv.zero lines found).")
        return

    named_modules = dict(model.named_modules())

    for base_name, zeroed_out_ch in base_to_zeroed_ch.items():
        # recupera il modulo C2fAttn
        c2f, parent, attr = _get_module_by_name(model, base_name)
        if not isinstance(c2f, C2fAttn):
            # alcune architetture hanno un path diverso; prova anche '<base>.0' ecc. se serve
            LOGGER.warning(f"Skip '{base_name}': not a C2fAttn.")
            continue

        attn: MaxSigmoidAttnBlock = c2f.attn
        proj = attn.proj_conv.conv  # Conv2d
        nh  = int(attn.nh)
        hc  = int(attn.hc)          # canali per head
        c_attn = proj.out_channels  # dovrebbe essere == nh*hc
        assert c_attn == nh * hc, f"Unexpected proj_conv.out={c_attn} vs nh*hc={nh*hc}."

        zeroed_out_ch = sorted(set(zeroed_out_ch))
        # Mappa canali -> head id
        heads = sorted(set([ch // hc for ch in zeroed_out_ch]))
        if not heads:
            continue

        # 2) calcola gli indici di out-channels da TENERE per proj_conv
        all_idx = torch.arange(c_attn)
        drop_mask = torch.zeros_like(all_idx, dtype=torch.bool)
        for h in heads:
            start = h * hc
            drop_mask[start:start+hc] = True
        keep_out_idx = all_idx[~drop_mask]

        # 3) ricostruisci proj_conv con meno out-channels
        new_proj = _rebuild_conv_out(proj, keep_out_idx)
        attn.proj_conv.conv = new_proj

        # 3.bis) ricostruisci anche la BN associata a proj_conv mantenendo lo stesso ordine dei canali tenuti
        old_bn = getattr(attn.proj_conv, 'bn', None)
        if isinstance(old_bn, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(len(keep_out_idx), eps=old_bn.eps, momentum=old_bn.momentum, affine=True, track_running_stats=True)
            with torch.no_grad():
                new_bn.weight.copy_(old_bn.weight[keep_out_idx])
                new_bn.bias.copy_(old_bn.bias[keep_out_idx])
                new_bn.running_mean.copy_(old_bn.running_mean[keep_out_idx])
                new_bn.running_var.copy_(old_bn.running_var[keep_out_idx])
            attn.proj_conv.bn = new_bn

        # 4) aggiorna nh, bias e scale di MaxSigmoidAttnBlock
        new_nh = nh - len(heads)
        assert new_nh >= 1, "Pruning would remove all heads — non supportato."
        # bias: shape (nh,)
        with torch.no_grad():
            if hasattr(attn, "bias") and isinstance(attn.bias, torch.nn.Parameter):
                bias_keep = torch.ones(nh, dtype=torch.bool)
                bias_keep[heads] = False
                attn.bias = torch.nn.Parameter(attn.bias.data[bias_keep].clone())
            # scale: (1, nh, 1, 1) o float 1.0
            if isinstance(attn.scale, torch.nn.Parameter):
                scale_keep = torch.ones(nh, dtype=torch.bool, device=attn.scale.device)
                scale_keep[heads] = False
                attn.scale = torch.nn.Parameter(attn.scale[:, scale_keep, :, :].clone())
            # aggiorna nh e hc
            attn.nh = new_nh
            attn.hc = hc  # invariato (abbiamo tolto head intere)

         # === Coupling verso gl/ec ===
        gl_in = getattr(attn.gl, 'in_features', None)

        if hasattr(attn, 'gl') and isinstance(attn.gl, nn.Linear) and gl_in == c_attn:
            # caso accoppiato (già presente): rimuovi colonne in gl.weight con keep_out_idx
            with torch.no_grad():
                attn.gl.weight = nn.Parameter(attn.gl.weight[:, keep_out_idx].contiguous())
                attn.gl.in_features = len(keep_out_idx)

        elif hasattr(attn, 'gl') and isinstance(attn.gl, nn.Linear) and gl_in is not None and gl_in != c_attn:
            # caso DECOUPLED: cerca un projector con IN==c_attn e OUT==gl_in
            proj_map = _find_projector_in(attn, c_attn, gl_in)
            if isinstance(proj_map, nn.Conv2d):
                # riduci i suoi IN in base a keep_out_idx
                new_proj_map = _rebuild_conv_in(proj_map, keep_out_idx)
                # sostituisci in-place
                for pname, pmod in attn.named_modules():
                    if pmod is proj_map:
                        # rimpiazza sull'oggetto padre
                        parent = attn
                        # trova parent e attr
                        parent_ref = attn
                        for part in pname.split('.')[:-1]:
                            parent_ref = getattr(parent_ref, part) if not part.isdigit() else parent_ref[int(part)]
                        last = pname.split('.')[-1]
                        setattr(parent_ref, last, new_proj_map)
                        break
            elif isinstance(proj_map, nn.Linear):
                new_lin = _rebuild_linear_in(proj_map, keep_out_idx)
                for pname, pmod in attn.named_modules():
                    if pmod is proj_map:
                        parent_ref = attn
                        for part in pname.split('.')[:-1]:
                            parent_ref = getattr(parent_ref, part) if not part.isdigit() else parent_ref[int(part)]
                        setattr(parent_ref, pname.split('.')[-1], new_lin)
                        break
            else:
                LOGGER.info(f"[DECOUPLED-NO-MAP] {base_name}: gl_in={gl_in}, c_attn={c_attn}. No projector found; keeping gl unchanged.")
                    
        # 5.bis) Propaga il taglio anche su gl (IN) ed ec (OUT) se sono direttamente accoppiati
        # gl: Linear(con in_features == c_attn) -> rimuoviamo le colonne corrispondenti ai canali tagliati
        if hasattr(attn, 'gl') and isinstance(attn.gl, nn.Linear) and getattr(attn.gl, 'in_features', None) == c_attn:
            with torch.no_grad():
                attn.gl.weight = nn.Parameter(attn.gl.weight[:, keep_out_idx].contiguous())
                attn.gl.in_features = len(keep_out_idx)
            # bias/out_features restano invariati qui; se altrove riduci out, gestisci separatamente

        # ec: Conv/Conv-BN con out_channels == c_attn -> rimuoviamo le stesse uscite
        if hasattr(attn, 'ec') and hasattr(attn.ec, 'conv') and getattr(attn.ec.conv, 'out_channels', None) == c_attn:
            ec_conv = attn.ec.conv
            new_ec_conv = _rebuild_conv_out(ec_conv, keep_out_idx)
            attn.ec.conv = new_ec_conv
            # se esiste BN in attn.ec, allineala sugli stessi canali
            ec_bn = getattr(attn.ec, 'bn', None)
            if isinstance(ec_bn, nn.BatchNorm2d):
                new_ec_bn = nn.BatchNorm2d(len(keep_out_idx), eps=ec_bn.eps, momentum=ec_bn.momentum, affine=True, track_running_stats=True)
                with torch.no_grad():
                    new_ec_bn.weight.copy_(ec_bn.weight[keep_out_idx])
                    new_ec_bn.bias.copy_(ec_bn.bias[keep_out_idx])
                    new_ec_bn.running_mean.copy_(ec_bn.running_mean[keep_out_idx])
                    new_ec_bn.running_var.copy_(ec_bn.running_var[keep_out_idx])
                attn.ec.bn = new_ec_bn

        # 5) ricostruisci cv2 riducendo gli in-channels sulla fetta 'attn'
        # cv2 input = concat di: 2*self.c  (split), n*self.c (n bottleneck), e self.c (attn)
        # indici assoluti della fetta attn sono gli ULTIMI self.c canali di cv2 input
        cv2_conv = c2f.cv2.conv
        total_in = cv2_conv.in_channels
        c = c2f.c  # (= self.c nel C2fAttn)
        attn_slice_start = total_in - c
        attn_slice_end   = total_in
        # dentro questa fetta (lunghezza c), vanno rimossi proprio i canali keep_out_idx complementari
        # ma attenzione: keep_out_idx è su [0..c_attn-1] e c_attn == c
        assert c_attn == c, f"Mismatch attn proj out {c_attn} vs C2fAttn.c {c}"

        # costruiamo indici completi da TENERE per cv2 in-channels
        attn_all_local = torch.arange(c)                  # [0..c-1] locale alla fetta
        attn_keep_local = torch.arange(c)[~drop_mask]     # stessi keep della proj_conv
        # indici assoluti:
        pre_keep = torch.arange(attn_slice_start)
        attn_keep_abs = attn_keep_local + attn_slice_start
        keep_in_idx = torch.cat([pre_keep, attn_keep_abs], dim=0)

        new_cv2 = _rebuild_conv_in(cv2_conv, keep_in_idx)
        c2f.cv2.conv = new_cv2

        # 6) Aggiorna anche c2f.c? NO: c2f.c è il canale "hidden" fissato all'inizio (usato per costruire i blocchi).
        #    A runtime abbiamo ridotto solo la parte della concat relativa all'attn, e ricablato cv2 di conseguenza.
        #    Tutte le shape interne rimangono coerenti perché la cat ora passa meno canali nell'ultimo pezzo
        #    e cv2 è stato aggiornato per attenderne di meno.

        LOGGER.info(f"[C2fAttn] Pruned heads {heads} in '{base_name}': "
                    f"proj_conv.out {c_attn}->{new_proj.out_channels}, "
                    f"cv2.in {total_in}->{new_cv2.in_channels}, nh {nh}->{new_nh}")

# Crea una cartella univoca tipo 'runs/prune_run_0', 'prune_run_1', ecc.
def get_unique_run_dir(base_dir="runs/detect/prune_run"):
    i = 0
    while Path(f"{base_dir}_{i}").exists():
        i += 1
    run_dir = Path(f"{base_dir}_{i}")
    run_dir.mkdir(parents=True)
    return run_dir

def build_zero_masks(path: Path):
    """
    returns
        masks = {
            "model.2._layers.8.weight": dict(kind='out', idx=tensor([...]), mask=Tensor mask),
            "model.5.cv1.conv.weight":   dict(kind='in',  idx=tensor([...]), mask=Tensor mask),
            ...
        }
    """
    line_re = re.compile(r'^(.*?)\|([a-z]+)\|\[(.*)\]$')
    masks = {}
    with path.open() as f:
        for line in f:
            m = line_re.match(line.strip())
            if not m:
                continue                                     # skip garbage
            layer, kind, idx_str = m.groups()
            idx = torch.tensor(ast.literal_eval(f'[{idx_str}]'), dtype=torch.long)
            # Ultralytics keeps the learnable weight tensor under ".weight"
            key = f'{layer}.weight'
            masks[key] = dict(kind=kind, idx=idx)
    return masks


def apply_zero_mask(param: torch.nn.Parameter, idx: torch.Tensor, kind: str):
    """
    Zero for different tensor ranks:
      Conv2d.weight:   (out, in, k, k)
      BatchNorm.weight/bias: (C,)
      Linear.weight:   (out, in)
      bias:            (out,)
    kind:
      'out' -> zero rows / out-channels
      'in'  -> zero cols / in-channels
      'bn'  -> zero entries in 1D vectors
    """
    with torch.no_grad():
        if param.ndim == 4:           # Conv2d weight
            if kind == 'out':
                param[idx, ...] = 0
            elif kind == 'in':
                param[:, idx, ...] = 0
            else:
                raise ValueError(kind)
        elif param.ndim == 2:         # Linear weight
            if kind == 'out':
                param[idx, :] = 0
            elif kind == 'in':
                param[:, idx] = 0
            else:
                raise ValueError(kind)
        elif param.ndim == 1:         # BN weight/bias, bias vectors
            # treat as 'bn' or generic 'out'
            param[idx] = 0
        else:
            raise NotImplementedError(f"Unsupported param.ndim={param.ndim} for masking.")


def make_hooks(model, masks):
    """
    Re-zeros masked components at every step and kills their gradients.
    masks: {param_name: {'kind': 'in'|'out'|'bn', 'idx': LongTensor}}
    """
    # ci serve per trovare rapidamente la bias associata a un certo weight
    param_dict = dict(model.named_parameters())

    for name, p in model.named_parameters():
        if name not in masks:
            continue
        info = masks[name]
        idx  = info['idx']
        kind = info['kind']

        # --- 1) Zero ORA il peso mascherato
        apply_zero_mask(p.data, idx, kind)

        # --- 2) (NOVITÀ) se esiste la bias corrispondente, zero anche lei ORA
        #     vale per kind 'out' e 'bn' (dimensione su out-channels)
        b = None
        if name.endswith(".weight") and kind in ("out", "bn"):
            bias_name = name[:-len(".weight")] + ".bias"
            b = param_dict.get(bias_name, None)
            if b is not None and b.ndim == 1:
                with torch.no_grad():
                    b.data[idx] = 0

        # --- 3) Hook sul grad del PESO: azzera il grad sugli indici mascherati
        def grad_hook_w(g, idx=idx, kind=kind, p_ref=p):
            if p_ref.ndim == 4:       # Conv2d
                if kind == 'out': g[idx, ...] = 0
                elif kind == 'in': g[:, idx, ...] = 0
            elif p_ref.ndim == 2:     # Linear
                if kind == 'out': g[idx, :] = 0
                elif kind == 'in': g[:, idx] = 0
            elif p_ref.ndim == 1:     # 1D vectors
                g[idx] = 0
            return g
        p.register_hook(grad_hook_w)

        # --- 4) (NOVITÀ) Hook sul grad della BIAS: azzera il grad sugli stessi indici
        if b is not None:
            def grad_hook_b(g, idx=idx):
                g[idx] = 0
                return g
            b.register_hook(grad_hook_b)

        # --- 5) Clamp persistente DOPO optimizer.step() (peso)
        def clamp_hook_w(param, idx=idx, kind=kind):
            apply_zero_mask(param.data, idx, kind)
        p._clamp_hook = clamp_hook_w  # mantieni un ref per evitare GC

        # --- 6) (NOVITÀ) Clamp persistente DOPO optimizer.step() (bias)
        if b is not None:
            def clamp_hook_b(param, idx=idx):
                param.data[idx] = 0
            b._clamp_hook = clamp_hook_b



# TAYLOR IMPORTANCE
def zero_c2fattn_heads_taylor(model, head_ratio: float, log_file: Path, eps: float = 1e-12):
    """
    Azzera intere head in MaxSigmoidAttnBlock usando Taylor importance.
    Richiede gradienti già popolati (W.grad non None), es. con 1-2 batch di backward senza step().
    """
    assert 0.0 <= head_ratio < 1.0, "head_ratio must be in [0,1)"
    lines = []

    for name, mod in model.named_modules():
        if isinstance(mod, C2fAttn):
            attn = mod.attn
            nh = getattr(attn, "nh", None)
            hc = getattr(attn, "hc", None)
            if nh is None or hc is None or nh <= 1 or hc <= 0:
                continue

            # numero teste da azzerare
            k = max(1, int(round(nh * head_ratio))) if head_ratio > 0 else 0
            if k == 0:
                continue

            W = attn.proj_conv.conv.weight     # [c2, in, k, k]
            G = W.grad
            if G is None:
                raise RuntimeError(
                    f"Taylor importance: gradients not found for {name}.attn.proj_conv.conv.weight. "
                    "Esegui un forward+backward (senza optimizer.step) per popolare .grad."
                )

            c2 = W.shape[0]
            assert c2 == nh * hc, f"Mismatch: expected {nh*hc} channels, got {c2}"

            # Taylor head importance: sum(|W * G|) su tutte le posizioni della testa
            head_imp = []
            with torch.no_grad():
                WG = (W * G).abs()  # [c2, in, k, k]
                for h in range(nh):
                    s = h * hc
                    e = s + hc
                    imp = WG[s:e, ...].sum().item()
                    head_imp.append((h, imp + eps))  # eps per stabilità

            head_imp.sort(key=lambda x: x[1])       # più piccola = meno importante
            heads_to_zero = [h for h, _ in head_imp[:k]]

            # indici out da azzerare (blocchi di hc)
            out_idx = []
            for h in heads_to_zero:
                out_idx.extend(range(h * hc, (h + 1) * hc))
            out_idx = torch.tensor(out_idx, dtype=torch.long, device=W.device)

            # azzera pesi e bias della proj_conv
            with torch.no_grad():
                W[out_idx, ...] = 0
                if attn.proj_conv.conv.bias is not None:
                    attn.proj_conv.conv.bias.data[out_idx] = 0

            # azzera anche BN associata alla proj_conv
            bn = getattr(attn.proj_conv, 'bn', None)
            if isinstance(bn, nn.BatchNorm2d):
                with torch.no_grad():
                    if bn.weight is not None: bn.weight.data[out_idx] = 0
                    if bn.bias   is not None: bn.bias.data[out_idx]   = 0
                    if bn.running_mean is not None: bn.running_mean.data[out_idx] = 0
                    if bn.running_var  is not None: bn.running_var.data[out_idx]  = 1
                lines.append(f"{name}.attn.proj_conv.bn|bn|{out_idx.tolist()}")

            # logging compatibile con il tuo parser
            lines.append(f"{name}.attn.proj_conv.conv|out|{out_idx.tolist()}")

            # opzionale: log accoppiamenti ec/gl solo se isomorfi (come nel tuo codice)
            ec_out = getattr(getattr(attn, "ec", None), "conv", None)
            ec_out = getattr(ec_out, "out_channels", None)
            gl_out = getattr(getattr(attn, "gl", None), "out_features", None)
            if ec_out == nh * hc and gl_out == nh * hc:
                lines.append(f"{name}.attn.ec.conv|out|{out_idx.tolist()}")
                lines.append(f"{name}.attn.gl|out|{out_idx.tolist()}")

    with log_file.open("a") as f:
        for ln in lines:
            f.write(ln + "\n")

#MAGNITUDE IMPORTANCE
def zero_c2fattn_heads(model, head_ratio: float, log_file: Path):
    """
    Azzera intere head in MaxSigmoidAttnBlock basandosi sulla magnitudine dei pesi.
    """
    assert 0.0 <= head_ratio < 1.0, "head_ratio must be in [0,1)"
    lines = []

    for name, mod in model.named_modules():
        if isinstance(mod, C2fAttn):
            attn = mod.attn
            nh = getattr(attn, "nh", None)
            hc = getattr(attn, "hc", None)
            if nh is None or hc is None or nh <= 1 or hc <= 0:
                continue

            # Numero di head da azzerare
            k = max(1, int(round(nh * head_ratio))) if head_ratio > 0 else 0
            if k == 0:
                continue

            # --- CALCOLO IMPORTANZA PER TESTA ---
            proj_weight = attn.proj_conv.conv.weight.data  # [c2, in_channels, k, k]
            c2 = proj_weight.shape[0]
            assert c2 == nh * hc, f"Mismatch: expected {nh*hc} channels, got {c2}"

            head_importance = []
            for h in range(nh):
                start = h * hc
                end   = start + hc
                # Calcola norma L2 di tutti i filtri relativi a questa head
                head_weight = proj_weight[start:end, ...]
                importance = torch.norm(head_weight, p=2).item()
                head_importance.append((h, importance))

            # --- ORDINA PER IMPORTANZA ---
            head_importance.sort(key=lambda x: x[1])  # crescente
            heads_to_zero = [h for h, _ in head_importance[:k]]

            # --- CREA GLI INDICI DEI CANALI DA AZZERARE ---
            out_idx = []
            for h in heads_to_zero:
                start = h * hc
                out_idx.extend(range(start, start + hc))
            out_idx = torch.tensor(out_idx, dtype=torch.long)

            # --- AZZERA I PESI ---
            with torch.no_grad():
                attn.proj_conv.conv.weight.data[out_idx, ...] = 0
                if attn.proj_conv.conv.bias is not None:
                    attn.proj_conv.conv.bias.data[out_idx] = 0

            # Zero anche la BN associata alla proj_conv (affine + running stats)
            bn = getattr(attn.proj_conv, 'bn', None)
            if isinstance(bn, nn.BatchNorm2d):
                with torch.no_grad():
                    # Affine
                    if bn.weight is not None:
                        bn.weight.data[out_idx] = 0
                    if bn.bias is not None:
                        bn.bias.data[out_idx] = 0
                    # Running stats
                    if bn.running_mean is not None:
                        bn.running_mean.data[out_idx] = 0
                    if bn.running_var is not None:
                        bn.running_var.data[out_idx] = 1
                # Log della BN in modo che le maschere la mantengano a zero durante il fine-tuning
                lines.append(f"{name}.attn.proj_conv.bn|bn|{out_idx.tolist()}")

            # --- LOGGING ---
            lines.append(f"{name}.attn.proj_conv.conv|out|{out_idx.tolist()}")

            ec_out = None
            if hasattr(attn, "ec") and isinstance(attn.ec, Conv):
                ec_out = attn.ec.conv.out_channels
            gl_out = getattr(getattr(attn, "gl", None), "out_features", None)
            if ec_out == nh * hc and gl_out == nh * hc:
                lines.append(f"{name}.attn.ec.conv|out|{out_idx.tolist()}")
                lines.append(f"{name}.attn.gl|out|{out_idx.tolist()}")

    with log_file.open("a") as f:
        for ln in lines:
            f.write(ln + "\n")



def prune(args):
    
    model = YOLOWorld(args.model) 

    pruning_cfg = yaml_load(check_yaml(args.cfg))
    batch_size = pruning_cfg['batch']
    pruning_cfg['data'] = "coco.yaml"
    finetune_epochs = args.finetune_epochs
    prune_rate = args.target_prune_rate
    head_prune_rate = args.head_prune_rate

    #create a file txt with target pruning rate
    with (RUN_DIR / "target_prune_rate.txt").open("w") as f:
        f.write(f"backbone {prune_rate}\n")
        f.write(f"head {head_prune_rate}\n")
    model.model.train()

    initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace
    for name, param in model.model.named_parameters():
        param.requires_grad = True

    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)
    macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    print(f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f}")
    
    # prune same ratio of filter based on initial size
    pruning_ratio = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps)

    for i in range(args.iterative_steps):

        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        for name, module in model.model.named_modules():
            if isinstance(module, (MaxSigmoidAttnBlock)):
                 for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True

        ignored_layers = []
        unwrapped_parameters = []

        #add first convolution to ignored layers model.model.model[0]._layers[1]._layers[0]
        ignored_layers.append(model.model.model[0]._layers[1]._layers[0])
        for m in model.model.modules():
            if  isinstance(m, (C2fAttn,)):# Concat, Conv, nn.Upsample,  )):
                ignored_layers.append(m)
    # 1) Ignora tutte le cv4 (testa contrastiva) e i loro parametri speciali, ma NON ignorare cv3 (le conv rimangono prunabili)---------FOR TAYLOR
            if isinstance(m, WorldDetect):
                # ignora ogni head cv4 (BNContrastiveHead o ContrastiveHead)
                for head in m.cv4:
                    ignored_layers.append(head)                   # esclude l'intero modulo cv4.*
                    # escludi anche parametri "unwrapped" (logit_scale, bias) se presenti
                    if hasattr(head, 'logit_scale'):
                        ignored_layers.append(head.logit_scale)   # nn.Parameter
                    if hasattr(head, 'bias'):
                        ignored_layers.append(head.bias)          # nn.Parameter
        #---------------------------------------------------------------------------------------------------------------FOR TAYLOR

        example_inputs = example_inputs.to(model.device)
        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs,
            importance=tp.importance.TaylorImportance(),
            iterative_steps=1,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
            global_pruning=True, 
        )

        # Test regularization
        output = model.model(example_inputs)
        (output[0].sum() + sum([o.sum() for o in output[1]])).backward()
        pruner.regularize(model.model)
        
       # pruner.step()
        with (RUN_DIR / "zeroed_params.txt").open("w") as zf:
            for group in pruner.step(interactive=True):
                for dep, idxs in group:
                    target_layer = dep.target.module
                    pruning_fn   = dep.handler

                    # decide IN/OUT/BATCHNORM and zero the weights
                    if pruning_fn in [tp.prune_conv_in_channels, tp.prune_linear_in_channels]:
                        target_layer.weight.data[:, idxs] *= 0
                        kind = "in"
                    elif pruning_fn in [tp.prune_conv_out_channels, tp.prune_linear_out_channels]:
                        target_layer.weight.data[idxs] *= 0
                        if target_layer.bias is not None:
                            target_layer.bias.data[idxs] *= 0
                        kind = "out"
                    elif pruning_fn in [tp.prune_batchnorm_out_channels]:
                        target_layer.weight.data[idxs] *= 0
                        target_layer.bias.data[idxs]   *= 0
                        kind = "bn"
                    else:
                        continue    # safety – shouldn’t reach

                    # ❶ write a concise log line:  layer-name | kind | zeroed-indices
                    layer_name = dep.target.name                      # e.g. "model.0._layers.3..."
                    zf.write(f"{layer_name}|{kind}|{idxs}\n")

                    # ❷ (optional) console feedback
                    LOGGER.info(f"Zeroed {len(idxs)} {kind}-channels in {layer_name}")

       # zero_c2fattn_heads(model.model, head_prune_rate, RUN_DIR / "zeroed_params.txt")
        zero_c2fattn_heads_taylor(model.model, head_prune_rate, RUN_DIR / "zeroed_params.txt")
        LOGGER.info(f"Applied head-zeroing to C2fAttn with ratio={head_prune_rate}")
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs.to(model.device))
        print(f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, ")
        
        # #save the pt model
        pruned_model_path = RUN_DIR / "zeroed_weights.pt"
        model.save(pruned_model_path)
        #save the state dict
        pruned_state_dict_path = RUN_DIR /"zeroed_weights_dict.pt"
        torch.save(model.model.state_dict(), pruned_state_dict_path)

        del pruner


    zeroed_model = YOLOWorld(str(RUN_DIR / "zeroed_weights.pt"))
    masks = build_zero_masks(RUN_DIR / "zeroed_params.txt")
    make_hooks(zeroed_model.model, masks)

    def _apply_masks_to_module(module, masks):
        """Riapplica maschere a pesi e bias di un `nn.Module` (no DDP wrapper)."""
        param_dict = dict(module.named_parameters())
        with torch.no_grad():
            for name, info in masks.items():
                if name not in param_dict:
                    continue
                p = param_dict[name]
                idx, kind = info["idx"], info["kind"]

                # peso
                apply_zero_mask(p.data, idx, kind)

                # bias associata (solo per 'out'/'bn')
                if kind in ("out", "bn") and name.endswith(".weight"):
                    bias_name = name[:-len(".weight")] + ".bias"
                    b = param_dict.get(bias_name, None)
                    if b is not None and b.ndim == 1:
                        b.data[idx] = 0


    def _clamp_masked_after_batch(trainer):
        # modello "vivo"
        _apply_masks_to_module(trainer.model, masks)
        # EMA (se esiste)
        ema_module = getattr(getattr(trainer, "ema", None), "ema", None)
        if ema_module is not None:
            _apply_masks_to_module(ema_module, masks)
    # Registra la callback Ultralytics (puoi fare via argomento o con add_callback)
    zeroed_model.add_callback("on_train_batch_end", _clamp_masked_after_batch)
    # opzionale: ulteriore sicurezza a fine epoca
    zeroed_model.add_callback("on_fit_epoch_end", _clamp_masked_after_batch)
    zeroed_model.train(data="coco.yaml", epochs=finetune_epochs, imgsz=640, batch=32, name= "zeroed_weights_finetune", project=str(RUN_DIR))


    with torch.no_grad():
        for name, p in zeroed_model.model.named_parameters():
            if name in masks:
                assert torch.allclose(p[masks[name]['idx']], torch.zeros_like(p[masks[name]['idx']])), \
                    f"{name} has non-zero masked weights!"
    print("All masked weights remain zero")


def prune_zeroed_weights(args):

    path = RUN_DIR / "zeroed_weights_finetune" / "weights" / "last.pt"
    #path = Path("runs/detect/prune_run_0/zeroed_weights_finetune/weights/last.pt")
    model = YOLOWorld(str(path))  # or any other yolov8 model

    model.model.train()
    for name, param in model.model.named_parameters():
        param.requires_grad = True
    example_inputs = torch.randn(1, 3, 640, 640).to(model.device)
    
    # prune same ratio of filter based on initial size
    pruning_ratio = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps)

    for i in range(args.iterative_steps):

        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        for name, module in model.model.named_modules():
            if isinstance(module, (MaxSigmoidAttnBlock)):
                 for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True

        ignored_layers = []
        unwrapped_parameters = []

   
        ignored_layers = []
        #add first convolution to ignored layers model.model.model[0]._layers[1]._layers[0]
        ignored_layers.append(model.model.model[0]._layers[1]._layers[0])
        for m in model.model.modules():
            if  isinstance(m, (C2fAttn,)): # Concat, Conv, nn.Upsample,  )):
                ignored_layers.append(m)
        

        example_inputs = example_inputs.to(model.device)
        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs,
            importance=tp.importance.GroupMagnitudeImportance(),  # L2 norm pruning,
            iterative_steps=1,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
            global_pruning=True, 
        )


        # Test regularization
        #output = model.model(example_inputs)
        #(output[0].sum() + sum([o.sum() for o in output[1]])).backward()
        #pruner.regularize(model.model)
        
        pruner.step()

        #PRUNE ALSO TRANSFORMER HEAD (C2FATTN LAYERS)

        log_file = RUN_DIR / "zeroed_params.txt"
        #log_file = Path("runs/detect/prune_run_0/zeroed_params.txt")
        structurally_prune_c2fattn_heads(model.model, log_file)
        fix_attn_invariants(model.model)                     
        fix_proj_bn_after_head_prune(model.model)

        # #save the pt model
        pruned_model_path = RUN_DIR / "pruned_model.pt"
        #pruned_model_path = Path("runs/detect/prune_run_0/pruned_model.pt")
        model.save(pruned_model_path)
        #save the state dict
        pruned_state_dict_path = RUN_DIR / "pruned_model_dict.pt"
       # pruned_state_dict_path = Path("runs/detect/prune_run_0/pruned_model_dict.pt")
        torch.save(model.model.state_dict(), pruned_state_dict_path)
        del pruner
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='ultralytics/cfg/default.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')
    parser.add_argument('--model', default='n-epoch100.pt', type=str, help='Path to the YOLOv8 model file')
    parser.add_argument('--iterative-steps', default=1, type=int, help='Total pruning iteration step')
    parser.add_argument('--target-prune-rate', default=0.7, type=float, help='Target pruning rate')
    parser.add_argument('--head-prune-rate', default=0.7, type=float, help='Target head pruning rate inside C2fAttn')
    parser.add_argument('--max-map-drop', default=0.2, type=float, help='Allowed maximum map drop after fine-tuning')
    parser.add_argument('--finetune-epochs', default=500, type=int, help='Number of fine-tuning epochs')

    args = parser.parse_args()

    #check if gpu is available
    # if not torch.cuda.is_available():
    #     raise RuntimeError("CUDA is not available. Please run this script on a machine with a GPU.")


    RUN_DIR = get_unique_run_dir()
    prune(args)
    #breakpoint()
    prune_zeroed_weights(args)


