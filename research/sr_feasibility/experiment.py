"""Synthetic feasibility study, NOT a P2 Pro measurement or production solver.

Known translations, known spatially invariant Gaussian effective PSF,
stationary scene, independent Gaussian noise, periodic boundaries. CGLS/CG
with quadratic gradient regularization. No registration time is included.
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from burst_sr import BurstSR

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(7)
h, w, s, n = 192, 256, 2, 16
H, W = h*s, w*s
yy, xx = np.mgrid[:H, :W].astype(np.float32)
truth = (3*np.exp(-((xx-220)**2+(yy-160)**2)/10000)).astype(np.float32)
truth[48:160, 48:224] += 5
truth[190:320, 260:440] += 3
# Sinusoidal gratings include frequencies above the LR Nyquist limit (0.5).
for j, frequency in enumerate((0.30, 0.45, 0.60, 0.70)):
    rows = slice(205+j*31, 229+j*31)
    truth[rows, 40:220] += 2 + 1.5*np.cos(2*np.pi*frequency*xx[rows,40:220]/s)
truth[70:145, 300:308] += 6
truth[70:145, 312:320] += 6
truth[70:145, 324:332] += 6
truth += 20

# Effective blur includes optical/sensor smoothing in this simplified model.
# It is not a measured PSF for the P2 Pro.
sigma_lr = 0.50
kernel = cv2.getGaussianKernel(9, sigma_lr*s).astype(np.float32).ravel()

def blur(a):
    p = len(kernel)//2
    padded = np.pad(a, ((p,p),(p,p)), mode='wrap')
    return cv2.sepFilter2D(padded, -1, kernel, kernel)[p:-p,p:-p]

def make_maps(offsets):
    maps = []
    gy, gx = np.mgrid[:h, :w].astype(np.float32)
    for dx, dy in offsets:
        # Frame displacement is positive dx,dy; sample the reference at x-dx.
        mx, my = s*(gx-dx), s*(gy-dy)
        ix, iy = np.floor(mx).astype(np.int32), np.floor(my).astype(np.int32)
        ax, ay = mx-ix, my-iy
        taps=[]
        for ox, oy, weight in ((0,0,(1-ax)*(1-ay)), (1,0,ax*(1-ay)),
                               (0,1,(1-ax)*ay), (1,1,ax*ay)):
            index=(((iy+oy)%H)*W+(ix+ox)%W).ravel()
            taps.append((index, weight.astype(np.float32).ravel()))
        maps.append(taps)
    return maps

offsets = np.vstack((np.zeros((1,2)), rng.uniform(-.85,.85,(n-1,2)))).astype(np.float32)
maps = make_maps(offsets)

def forward(x, mapping=maps):
    b = blur(x.reshape(H,W)).ravel()
    return np.stack([sum(b[idx]*wt for idx,wt in taps) for taps in mapping])

def adjoint(y, mapping=maps):
    acc = np.zeros(H*W, np.float64)
    for vals,taps in zip(y,mapping):
        for idx,wt in taps:
            acc += np.bincount(idx, weights=vals*wt, minlength=H*W)
    return blur(acc.astype(np.float32).reshape(H,W)).ravel()

def lap(x):
    a=x.reshape(H,W)
    return (4*a-np.roll(a,1,0)-np.roll(a,-1,0)-np.roll(a,1,1)-np.roll(a,-1,1)).ravel()

dot=lambda a,b: float(np.einsum('i,i->',a.ravel(),b.ravel(),dtype=np.float64))
test_x=rng.standard_normal(H*W).astype(np.float32)
test_y=rng.standard_normal((n,h*w)).astype(np.float32)
lhs,rhs=dot(forward(test_x),test_y),dot(test_x,adjoint(test_y))
adjoint_error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1)
assert adjoint_error < 1e-5, adjoint_error

noise_sigma=.05
observations=forward((truth-20).ravel())+rng.normal(0,noise_sigma,(n,h*w)).astype(np.float32)
frames=(observations.reshape(n,h,w)+20).astype(np.float32)
sr=BurstSR((h,w),scale=s,max_frames=n)
# Oracle registration supplied to both algorithms to isolate reconstruction.
sr._ref=frames[0].copy()
sr._buffer=deque([(f,float(dx),float(dy)) for f,(dx,dy) in zip(frames,offsets)],maxlen=n)
t0=time.perf_counter(); existing=sr.get_hr(); existing_s=time.perf_counter()-t0
t0=time.perf_counter(); hq=sr.get_hq(); hq_s=time.perf_counter()-t0
single=sr._remap(frames[0],0,0)

def solve(mapping=maps, lam=.1, iterations=25):
    b=adjoint(observations,mapping)
    op=lambda x: adjoint(forward(x,mapping),mapping)+lam*lap(x)
    x=(single-20).ravel().copy()
    r=b-op(x); p=r.copy(); rr=dot(r,r)
    for _ in range(iterations):
        ap=op(p)
        alpha=rr/max(dot(p,ap),1e-30)
        x+=alpha*p; r-=alpha*ap
        next_rr=dot(r,r)
        p=r+(next_rr/max(rr,1e-30))*p
        rr=next_rr
    return x.reshape(H,W)+20

t0=time.perf_counter(); inverse=solve(); inverse_s=time.perf_counter()-t0
badmaps=make_maps(offsets+rng.normal(0,.10,offsets.shape).astype(np.float32))
t0=time.perf_counter(); bad_inverse=solve(badmaps); bad_s=time.perf_counter()-t0

crop=np.s_[12:-12,12:-12]
def metrics(a):
    err=(a-truth)[crop]
    flat=(a-truth)[18:38,350:450]
    return {'rmse_synthetic_units':float(np.sqrt(np.mean(err**2))),
            'flat_error_std':float(flat.std())}
regularization_sweep={}
for strength in (.003,.01,.03,.1):
    candidate=inverse if strength==.1 else solve(lam=strength)
    regularization_sweep[str(strength)]=metrics(candidate)

result={'model':'synthetic, known shifts and PSF; not camera validation',
        'lr_shape':[h,w],'hr_shape':[H,W],'frames':n,'iterations':25,
        'effective_gaussian_psf_sigma_lr':sigma_lr,'input_noise_sigma':noise_sigma,
        'regularization_lambda':.1,'regularization_sweep':regularization_sweep,
        'adjoint_relative_error':adjoint_error,
        'timing_seconds':{'existing_live':existing_s,'existing_hq':hq_s,
                          'inverse_known_shifts':inverse_s,'inverse_wrong_shifts':bad_s},
        'metrics':{name:metrics(a) for name,a in [('single_bilinear',single),('existing_live',existing),
            ('existing_hq',hq),('inverse_known_shifts',inverse),('inverse_shift_error_0.1px',bad_inverse)]}}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
panels=[('Синтетический оригинал',truth),('Один кадр: интерполяция',single),
        ('Текущий BurstSR: сдвиги известны',existing),('Реконструкция: сдвиги и PSF известны',inverse)]
montage=Image.new('RGB',(2*W,2*(H+40)+80),'white')
draw=ImageDraw.Draw(montage)
try: font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',19)
except OSError: font=ImageFont.load_default()
for i,(label,a) in enumerate(panels):
    x=(i%2)*W; y=(i//2)*(H+40)
    u=np.clip((a-20)/8*255,0,255).astype(np.uint8)
    montage.paste(Image.fromarray(u).convert('RGB'),(x,y+40))
    draw.text((x+12,y+9),label,fill='black',font=font)
draw.text((12,2*(H+40)+8),'Синтетическая проверка: 16 кадров, сетка 2×; единая шкала яркости.',fill='black',font=font)
draw.text((12,2*(H+40)+37),'Идеально известные сдвиги и размытие. Это не съёмка P2 Pro.',fill='black',font=font)
montage.save(OUT/'comparison.png')
print(json.dumps(result,indent=2))
