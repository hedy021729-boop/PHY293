#!/usr/bin/env python3
"""Michelson interferometry lab analysis (green laser, prism, aluminium).

Run:
    python interferometry_analysis.py --data-dir . --out results

Required CSV file names (as uploaded):
    interferometry - 1.csv
    interferometry - 2.csv
    interferometry - 3.csv

Python dependencies: numpy, scipy, matplotlib.

Statistical conventions:
* Supplied +/- half-resolution bounds are converted to 1-sigma assuming
  independent uniform rounding within each interval (divide by sqrt(3)).
* +/-0.5 fringe is an estimated 1-sigma count-reading uncertainty, not a
  guarantee that no whole fringes were missed. Missed counts are assessed
  separately in residuals and the discussion.
* A shared reference uncertainty is NOT treated as an independent error for
  every observation. With a free intercept, a common additive x offset cannot
  change the fitted slope. The reference uncertainty is nevertheless included
  in the plotted horizontal error bars, and prism reference bias is explored
  separately with a sensitivity check.
* Primary fits use independent errors in BOTH x and y, implemented with a
  least-squares fit to latent (adjusted) x positions. Parameter covariance is
  calculated from the Jacobian using absolute input 1-sigma uncertainties.
  Reduced chi-square is obtained from the corresponding orthogonal weighted
  residuals, with DOF = n_observations - n_physical_parameters.
* Course-aligned, free-intercept ordinary least-squares fits and scatter-based
  parameter standard errors are also calculated for comparison. This is
  especially important when interpreting the 'Uncertainties in linear fit'
  handout, which uses OLS rather than fitting x-uncertainty.
* At T=T0, N=0 is a chosen baseline, not an independent datum. It is shown
  as an unfilled symbol but excluded from aluminium regression.

Please read the automatically generated 'analysis_notes.txt' before reporting
these as final experimental uncertainties. Unquantified effects remain.
"""

from __future__ import annotations
import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------- EDITABLE INPUTS ---------------------------- #
# Maximal reading deviations (not Gaussian 1-sigma values)
MICROMETER_BOUND_UM = 0.5
ANGLE_READING_BOUND_DEG = 0.5
TEMP_READING_BOUND_C = 0.05
CALIPER_READING_BOUND_MM = 0.005
# Uncertainty estimated for reproducibility of the first/last fringe phase.
FRINGE_COUNT_SIGMA = 1.0
# Reference-angle bias explored as a systematic sensitivity interval
PRISM_REFERENCE_BIAS_BOUND_DEG = 0.5
# The 3 green runs and prism +/- directions are taken as distinct runs.
# Do NOT add an arbitrary (0,0) to the green/prism datasets.
# Aluminium T0,N0 is a chosen reference, not an independent fit point.

C_GREEN = '#2476a9'
C_RED = '#c45441'
C_GOLD = '#a37a23'
C_FIT = '#2a2c34'
SIGMA_UNIFORM_FACTOR = 1 / np.sqrt(3.0)


def read_numeric_rows(path: Path, columns: tuple[int, ...]) -> np.ndarray:
    """Skip metadata/headers, accept only complete numeric rows in columns."""
    out = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.reader(handle):
            try:
                numbers = [float(row[k].strip()) for k in columns]
            except (IndexError, ValueError):
                continue
            if np.all(np.isfinite(numbers)):
                out.append(numbers)
    if not out:
        raise ValueError(f'Could not find numeric rows in {path}')
    return np.asarray(out, dtype=float)


def fit_course_ols(x, y, origin=False):
    """Handout method: scatter-based parameter standard errors from OLS."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = x[:, None] if origin else np.column_stack((x, np.ones_like(x)))
    p, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ p
    dof = len(x) - len(p)
    rss = float(np.sum((y-pred)**2))
    cov = np.linalg.inv(A.T @ A) * (rss / dof) if dof > 0 else np.full((len(p),len(p)),np.nan)
    return {'p': p, 's': np.sqrt(np.diag(cov)), 'rss': rss,
            'r2': 1 - rss / np.sum((y-y.mean())**2), 'dof': dof}


def fit_xy_errors(x, y, sx, sy, model, initial):
    """Errors-in-both-axes least squares with latent true x_i.

    Parameter vector = [physical parameters, true x_1, ..., true x_n].
    Minimize sum(((x_true-x_obs)/sx)^2 + ((y-f(x_true,p))/sy)^2).
    Because we fit n latent x parameters, DOF = 2*n-(n+n_phys) = n-n_phys.
    Absolute parameter covariance is inverse(J.T@J) at the solution (1-sigma
    input uncertainties, Gauss-Newton/local linear approximation).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sx = np.broadcast_to(sx, x.shape).astype(float)
    sy = np.broadcast_to(sy, x.shape).astype(float)
    assert np.all(sx > 0) and np.all(sy > 0)
    n = len(x)
    npar = len(initial)

    def residuals(z):
        p, xx = z[:npar], z[npar:]
        return np.r_[(xx-x)/sx, (model(xx, p)-y)/sy]

    result = least_squares(residuals, np.r_[initial, x],
                           method='trf', xtol=1e-12, ftol=1e-12,
                           gtol=1e-12, max_nfev=20000, x_scale='jac')
    if not result.success and result.optimality > 1e-4:
        raise RuntimeError(f'Fit failed: {result.message}')
    p = result.x[:npar]
    J = result.jac
    covariance_full = np.linalg.pinv(J.T @ J)
    covariance_p = covariance_full[:npar, :npar]
    sigma_p = np.sqrt(np.maximum(0, np.diag(covariance_p)))
    pred = model(x, p)  # vertical residual at displayed x, not ODR adjusted x
    ordinary_rss = np.sum((y-pred)**2)
    var_tot = np.sum((y-y.mean())**2)
    ndof = n-npar
    chi2 = float(np.sum(result.fun**2))
    return {'p':p, 's':sigma_p, 'cov':covariance_p,
            'x_true':result.x[npar:], 'pred':pred,
            'residual':y-pred, 'chi2':chi2, 'dof':ndof,
            'redchi2':chi2/ndof, 'r2':1-ordinary_rss/var_tot,
            'fit_success':result.success}


def linear(x, p): return p[0]*x + p[1]
def through_origin_square(theta, p): return p[0]*theta**2


def get_linear_fit(x,y,sx,sy):
    ols = fit_course_ols(x,y)
    adjusted = fit_xy_errors(x,y,sx,sy,linear,ols['p'])
    adjusted['ols'] = ols
    adjusted['model'] = 'linear'
    return adjusted


def get_prism_fit(theta,y,sigma_theta,sy):
    init = fit_course_ols(theta**2,y,origin=True)['p']
    adjusted = fit_xy_errors(theta,y,sigma_theta,sy,through_origin_square,init)
    adjusted['ols'] = fit_course_ols(theta**2,y,origin=True)
    adjusted['free_ols'] = fit_course_ols(theta**2,y,origin=False)
    adjusted['model'] = 'square'
    return adjusted


def styled_figure(title, xlabel, ylabel, fit_label, x, y, xerr, yerr,
                  residual_yerr, curve_fn, results, color, path, xlimit=None,
                  reference=None, annotation=None):
    """Single series, aligned main and residual axes, PDF + PNG."""
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11.7,
        'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'savefig.facecolor':'white'})
    fig, (top, bottom) = plt.subplots(2,1,figsize=(8.2,6.9), sharex=True,
                                      gridspec_kw={'height_ratios':[2.4,1], 'hspace':0.08})
    top.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o',ms=4.6,
                 elinewidth=0.85,capsize=2.3,capthick=0.8,
                 color=color,ecolor='#6b7480',label='Observed data',zorder=4)
    xlo,xhi = (np.min(x),np.max(x))
    margin = 0.05*(xhi-xlo) if xhi>xlo else 1
    xx = np.linspace(xlo-margin,xhi+margin,350)
    top.plot(xx,curve_fn(xx),lw=1.8,c=C_FIT,label=fit_label,zorder=2)
    if reference is not None:
        top.plot(reference[0],reference[1],marker='o',mec='#4e5b66',
                 mfc='white',markersize=6,linestyle='None',
                 label='Defined zero (not fitted)',zorder=5)
    top.set_ylabel(ylabel)
    top.set_title(title,weight='bold',pad=12)
    top.grid(alpha=0.16)
    top.legend(frameon=False,fontsize=10,loc='upper left')
    if annotation:
        top.text(.98,.04,annotation,transform=top.transAxes,
                 fontsize=10,va='bottom',ha='right',linespacing=1.4,
                 bbox={'facecolor':'white','edgecolor':'#dddddd','alpha':.92,'boxstyle':'round,pad=.6'})

    res = y-curve_fn(x)
    bottom.errorbar(x,res,xerr=xerr,yerr=residual_yerr,fmt='o',ms=4.3,
                    elinewidth=0.85,capsize=2.3,color=color,ecolor='#6b7480')
    bottom.axhline(0,color=C_FIT,lw=1.15)
    bottom.grid(alpha=0.16)
    bottom.set_ylabel('Residual (fringes)')
    bottom.set_xlabel(xlabel)
    rmax = max(np.max(np.abs(res)+residual_yerr)*1.18,0.5)
    bottom.set_ylim(-rmax,rmax)
    if xlimit:
        bottom.set_xlim(*xlimit)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),bbox_inches='tight',dpi=180)
    plt.close(fig)


def value_summary(fit):
    return f"m = {fit['p'][0]:.4g} +/- {fit['s'][0]:.2g}\n" + \
           (f"b = {fit['p'][1]:.4g} +/- {fit['s'][1]:.2g}\n" if len(fit['p'])>1 else 'b = 0 (constrained)\n') + \
           f"R² = {fit['r2']:.4f}; χ²ν = {fit['redchi2']:.3g}"


def append_fit_record(rows,name,fit,property_name,property_value,property_uncertainty):
    rows.append({'dataset':name, 'n':fit['dof']+len(fit['p']),
        'slope':fit['p'][0], 'sigma_slope':fit['s'][0],
        'intercept':fit['p'][1] if len(fit['p'])>1 else 0.0,
        'sigma_intercept':fit['s'][1] if len(fit['p'])>1 else 0.0,
        'OLS_slope':fit['ols']['p'][0],
        'OLS_slope_SE_handout':fit['ols']['s'][0],
        'OLS_intercept':fit['ols']['p'][1] if len(fit['ols']['p'])>1 else 0.0,
        'R_squared':fit['r2'], 'chi_squared':fit['chi2'],
        'degrees_freedom':fit['dof'], 'reduced_chi_squared':fit['redchi2'],
        property_name: property_value, f'uncertainty_{property_name}':property_uncertainty})


def main(data_dir, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    green = read_numeric_rows(data_dir/'interferometry - 1.csv',(0,1,2,3))
    prism = read_numeric_rows(data_dir/'interferometry - 2.csv',(0,1,2,3,4))
    aluminium = read_numeric_rows(data_dir/'interferometry - 3.csv',(0,1,2,3))
    # All +/- half-step resolutions interpreted as uniform bounds. Fringe is
    # independently justified as an estimated one-sigma uncertainty.
    sm = MICROMETER_BOUND_UM*SIGMA_UNIFORM_FACTOR
    sa = np.deg2rad(ANGLE_READING_BOUND_DEG*SIGMA_UNIFORM_FACTOR)
    st = TEMP_READING_BOUND_C*SIGMA_UNIFORM_FACTOR
    slength = CALIPER_READING_BOUND_MM*SIGMA_UNIFORM_FACTOR
    sn = FRINGE_COUNT_SIGMA
    rows = []
    green_fits = []
    xg = green[:,0]
    for j in range(3):
        yg = green[:,j+1]
        fit = get_linear_fit(xg,yg,sm,sn)
        green_fits.append(fit)
        slope, slope_sig = fit['p'][0], fit['s'][0]
        wave_nm = 2000/slope
        s_wave_nm = 2000*slope_sig/slope**2
        append_fit_record(rows,f'Green trial {j+1}',fit,'wavelength_nm',wave_nm,s_wave_nm)
        plot_note = value_summary(fit)+f'\nλ = ({wave_nm:.1f} +/- {s_wave_nm:.1f}) nm'
        styled_figure(f'Green laser · trial {j+1}',r'Mirror displacement Δx (µm)',
            'Cumulative fringe count N',f'Fit (slope={slope:.3f} fringes/µm)',
            xg,yg,np.full(len(xg),np.hypot(sm,sm)),np.full(len(yg),sn),
            np.full(len(yg),np.hypot(sn,slope*sm)),
            lambda xx,ff=fit: linear(xx,ff['p']), fit,C_GREEN if j==0 else [C_RED,C_GOLD][j-1],
            out_dir/f'green_trial_{j+1}',annotation=plot_note)

    # Combine green trials by mean of three independently fitted wavelengths.
    # Trial-to-trial SEM includes real run variation; the formal propagated
    # average of individual fit errors is reported separately, not quadrature
    # added (which could count the same noise twice).
    wave_vals = np.array([2000/f['p'][0] for f in green_fits])
    wave_sig_fit = np.array([2000*f['s'][0]/f['p'][0]**2 for f in green_fits])
    wave_mean = float(wave_vals.mean())
    wave_sem = float(wave_vals.std(ddof=1)/np.sqrt(len(wave_vals)))
    wave_formal_mean = float(np.sqrt(np.sum(wave_sig_fit**2))/len(wave_vals))
    # Conservative working uncertainty for later calculations; not a formally
    # combined 1-sigma confidence interval. Both components reported.
    wave_work_sigma = max(wave_sem,wave_formal_mean)
    # Draw combined green trials using their own fitted lines so visible
    # intercept differences aren't hidden by a forced shared zero.
    plt.rcParams.update({'pdf.fonttype':42})
    fig,(a,b)=plt.subplots(2,1,figsize=(8.2,6.9),sharex=True,
        gridspec_kw={'height_ratios':[2.4,1],'hspace':0.08})
    colors=[C_GREEN,C_RED,C_GOLD]
    xx=np.linspace(4,26,200)
    for j,fit in enumerate(green_fits):
        y=green[:,j+1]
        a.errorbar(xg,y,xerr=np.hypot(sm,sm),yerr=sn,fmt='o',ms=4,
            capsize=2,elinewidth=.7,ecolor='#8c94a0',color=colors[j],label=f'Trial {j+1}')
        a.plot(xx,linear(xx,fit['p']),color=colors[j],lw=1.5)
        b.errorbar(xg,y-linear(xg,fit['p']),xerr=np.hypot(sm,sm),
            yerr=np.hypot(sn,fit['p'][0]*sm),fmt='o',ms=4,
            capsize=2,elinewidth=.7,color=colors[j],ecolor='#8c94a0')
    a.set_title('Cumulative fringe count (N) vs. Mirror displacement (µm)',weight='bold')
    a.set_ylabel('Cumulative fringe count N')
    a.legend(frameon=False)
    a.grid(alpha=.16)
    b.axhline(0,color=C_FIT,lw=1.1)
    b.set_ylabel('Residual (fringes)')
    b.set_xlabel('Mirror displacement Δd (µm)')
    b.grid(alpha=.16)
    b.set_xlim(3,27)
    fig.savefig(out_dir/'green_comparison.pdf',bbox_inches='tight')
    fig.savefig(out_dir/'green_comparison.png',bbox_inches='tight',dpi=180)
    plt.close(fig)

    # Prism: 134° reference is the experiment's nominal reference. Individual
    # angles are in degrees in first CSV column. Fit each direction separately.
    theta = np.deg2rad(prism[:,0])
    u = theta**2
    directions = [('+',prism[:,3],1,C_GREEN),('-',prism[:,4],-1,C_RED)]
    prism_fits = []
    t_nm = 7_790_000.0
    sigma_t_nm = slength*1_000_000.0
    for label,yp,sign,color in directions:
        fit = get_prism_fit(theta,yp,sa,sn)
        prism_fits.append(fit)
        slope,ss=fit['p'][0],fit['s'][0]
        n_index=1/(1-slope*wave_mean/t_nm)
        # Algebraic propagation, assuming slope and wavelength uncertainties
        # independent; shared reference-angle bias handled separately below.
        n_sig=n_index**2*np.sqrt((wave_mean/t_nm*ss)**2+
                   (slope/t_nm*wave_work_sigma)**2+
                   (slope*wave_mean/t_nm**2*sigma_t_nm)**2)
        append_fit_record(rows,f'Prism 134{label}',fit,'refractive_index',n_index,n_sig)
        xerr=2*theta*np.hypot(sa,sa)  # delta(theta²), individual and shared resolution
        ryerr=np.sqrt(sn**2+(2*slope*theta*sa)**2) # only independent piece
        note=value_summary(fit)+f'\nn = {n_index:.3f} +/- {n_sig:.3f}'
        styled_figure(f'Plastic prism · 134°{label}',r'Squared angular displacement θ² (rad²)',
           'Cumulative fringe count N',f'N = ({slope:.0f}) θ²',
           u,yp,xerr,np.full(len(yp),sn),ryerr,
           lambda xx,mm=slope:mm*xx,fit,color,
           out_dir/f'prism_134_{"plus" if sign==1 else "minus"}',annotation=note)

    # Combined comparison of theoretical same-slope lines; don't silently
    # assume the two directions are statistically consistent.
    fig,(a,b)=plt.subplots(2,1,figsize=(8.2,7),sharex=True,
        gridspec_kw={'height_ratios':[2.4,1],'hspace':0.08})
    xx=np.linspace(0,u.max()*1.05,300)
    for (lab,yp,sign,color),f in zip(directions,prism_fits):
        m=f['p'][0]
        a.errorbar(u,yp,xerr=2*theta*np.hypot(sa,sa),yerr=sn,
                   fmt='o',capsize=2,ms=4,ecolor='#8e96a0',color=color,label=f'134°{lab}')
        a.plot(xx,m*xx,lw=1.5,color=color,label=f'134°{lab} fit')
        b.errorbar(u,yp-m*u,xerr=2*theta*np.hypot(sa,sa),
                   yerr=np.sqrt(sn**2+(2*m*theta*sa)**2),
                   fmt='o',capsize=2,color=color,ms=4,ecolor='#8e96a0')
    a.set_title('Plastic prism · opposite rotation directions',weight='bold')
    a.set_ylabel('Cumulative fringe count N');a.legend(frameon=False,ncol=2,fontsize=10)
    a.grid(alpha=.16);b.grid(alpha=.16)
    b.axhline(0,color=C_FIT,lw=1.1)
    b.set_ylabel('Residual (fringes)')
    b.set_xlabel('Squared angular displacement θ² (rad²)')
    fig.savefig(out_dir/'prism_comparison.pdf',bbox_inches='tight')
    fig.savefig(out_dir/'prism_comparison.png',bbox_inches='tight',dpi=180)
    plt.close(fig)

    # A transparent, optional diagnostic: reference offset δ causes opposite
    # signed linear terms: N_± = m*theta² ∓ 2*m*δ*theta. Fit ordinary least squares
    # to BOTH directions, *not* used to rewrite the measured counts.
    theta2=np.tile(theta,2)
    ys=np.r_[prism[:,3],prism[:,4]]
    s=np.r_[np.ones(len(theta)),-np.ones(len(theta))]
    M=np.column_stack((theta2**2,s*theta2))
    coeff,*_=np.linalg.lstsq(M,ys,rcond=None)
    m_joint,k_joint=coeff
    ref_shift_deg = float(np.rad2deg(-k_joint/(2*m_joint)))
    joint_n=1/(1-m_joint*wave_mean/t_nm)
    # Range from applying an assumed reference offset δ to the physically
    # correct change-in-count model N = m*(theta² - 2*sign*theta*δ).
    sensitivity=[]
    for deg in [-PRISM_REFERENCE_BIAS_BOUND_DEG,0,PRISM_REFERENCE_BIAS_BOUND_DEG]:
        d=np.deg2rad(deg)
        eff=np.r_[theta**2-2*theta*d,theta**2+2*theta*d]
        slope0=float(eff@ys/(eff@eff))
        sensitivity.append((deg,slope0,1/(1-slope0*wave_mean/t_nm)))

    # Aluminium. (deltaT,N)=(0,0) is baseline definition and excluded from fit.
    # Temp differences T_i-T0 include independent T_i and shared T0 uncertainty.
    al=aluminium[1:]
    ta=al[:,2]
    ya=al[:,3]
    af=get_linear_fit(ta,ya,st,sn)
    ma,sma=af['p'][0],af['s'][0]
    L0_mm=89.12
    alpha=ma*(wave_mean*1e-6)/(2*L0_mm)  # nm -> mm
    alpha_sig=abs(alpha)*np.sqrt((sma/ma)**2+
        (wave_work_sigma/wave_mean)**2+(slength/L0_mm)**2)
    append_fit_record(rows,'Aluminium expansion',af,'alpha_per_C',alpha,alpha_sig)
    anote=value_summary(af)+f'\nα = ({alpha*1e6:.2f} +/- {alpha_sig*1e6:.2f}) ×10⁻⁶ /°C'
    styled_figure('Aluminium · thermal expansion',r'Temperature rise ΔT (°C)',
        'Cumulative fringe count N','Linear errors-in-variables fit',
        ta,ya,np.full(len(ta),np.hypot(st,st)),np.full(len(ya),sn),
        np.full(len(ya),np.hypot(sn,ma*st)),
        lambda xx,ff=af:linear(xx,ff['p']),af,C_GREEN,
        out_dir/'aluminium_expansion',reference=(0,0),annotation=anote)

    # Summary table easy to paste into reports / spreadsheets.
    out_keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out_dir/'fit_results.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=out_keys);w.writeheader();w.writerows(rows)

    # Sample calculation values from the actual fitted data, LaTeX-ready.
    g0=green_fits[0];pr=prism_fits[0]
    n0=rows[3]['refractive_index']
    tex = r'''\section*{Appendix: Sample data analysis and uncertainty propagation}
\noindent All values below use the reported uncertainty convention: instrument
half-resolution bounds are converted to standard uncertainties by dividing by $\sqrt{3}$,
assuming uniform rounding errors. The uncertainty of locating fringes is assumed to be
$\sigma_N=0.5$ fringe. Common reference errors are not independent observations.

\subsection*{Green laser: wavelength from fitted slope}
\begin{align*}
N &= m_\lambda\Delta x+b,\\
\lambda &=\frac{2000}{m_\lambda}\;\mathrm{nm},\\
\sigma_\lambda &=\left|\frac{\partial\lambda}{\partial m_\lambda}\right|\sigma_m
=\frac{2000\sigma_m}{m_\lambda^2}.
\end{align*}
'''
    tex += (f"For green trial 1, $m_\\lambda={g0['p'][0]:.5f}\\pm{g0['s'][0]:.5f}$ fringes/$\\mu$m, hence\n"
            f"$\\lambda_1={2000/g0['p'][0]:.2f}\\pm{2000*g0['s'][0]/g0['p'][0]**2:.2f}$ nm.\n\n")
    tex += r'''\subsection*{Measurement uncertainty in mirror displacement}
If each micrometer reading has a maximum rounding error of $0.5\,\mu$m,
$\sigma_{x_i}=0.5/\sqrt{3}\,\mu$m. For a difference
$\Delta x_i=x_i-x_0$, the standard uncertainty of a single difference is
\[
\sigma_{\Delta x}=\sqrt{\sigma_{x_i}^2+\sigma_{x_0}^2}
=\sqrt{2}\frac{0.5}{\sqrt3}\,\mu\mathrm m.
\]
The same $x_0$ is shared by the data, so the point-to-point covariance is not zero.
A free intercept absorbs a constant reference offset when estimating the slope.

\subsection*{Angular transformation and refractive index}
\begin{align*}
\theta &= \frac{\pi}{180}\theta_{\mathrm{deg}},\qquad u=\theta^2,\\
\sigma_u&=\left|\frac{du}{d\theta}\right|\sigma_\theta=2|\theta|\sigma_\theta,\\
n&=\frac{1}{1-m_n\lambda/t}.
\end{align*}
'''
    tex += (f"At $\\theta=1^\\circ$, $u={(np.deg2rad(1)**2):.7f}$ rad$^2$ and with a single-angle\n"
            f"standard uncertainty of ${sa:.7f}$ rad, $\\sigma_u={2*np.deg2rad(1)*sa:.7f}$ rad$^2$.\n\n")
    tex += r'''For independent slope, wavelength, and thickness uncertainties,
\[
\sigma_n=n^2\sqrt{\left(\frac{\lambda\sigma_m}{t}\right)^2+
\left(\frac{m\sigma_\lambda}{t}\right)^2+
\left(\frac{m\lambda\sigma_t}{t^2}\right)^2}.
\]
'''
    tex += (f"For the 134$^\\circ$+ direction $m_n={pr['p'][0]:.3f}\\pm{pr['s'][0]:.3f}$ rad$^{{-2}}$,\n"
            f"$t={t_nm:.0f}$ nm, and $\\lambda={wave_mean:.3f}$ nm. This gives $n={n0:.4f}$ and\n"
            f"$\\sigma_n={rows[3]['uncertainty_refractive_index']:.4f}$. The shared-reference-angle\n"
            "bias is an additional possible systematic error, treated separately.\n\n")
    tex += r'''\subsection*{Temperature difference and aluminium expansion}
\begin{align*}
\Delta T_i &= T_i-T_0,\\
\sigma_{\Delta T_i}&=\sqrt{\sigma_{T_i}^2+\sigma_{T_0}^2},\\
\alpha &= \frac{m_\alpha\lambda}{2L_0},\\
\frac{\sigma_\alpha}{|\alpha|} &=
\sqrt{\left(\frac{\sigma_{m_\alpha}}{m_\alpha}\right)^2+
\left(\frac{\sigma_\lambda}{\lambda}\right)^2+
\left(\frac{\sigma_{L_0}}{L_0}\right)^2}.
\end{align*}
'''
    tex += (f"At $T=27.0^\\circ$C, $T_0=25.2^\\circ$C and each reading has\n"
            f"$\\sigma_T=0.05/\\sqrt3$ C, the difference is $\\Delta T=1.8^\\circ$C with\n"
            f"$\\sigma_{{\\Delta T}}={np.hypot(st,st):.4f}^\\circ$C.\n"
            f"The fitted slope is $m_\\alpha={ma:.5f}\\pm{sma:.5f}$ fringes/$^\\circ$C,\n"
            f"giving $\\alpha=({alpha*1e6:.3f}\\pm{alpha_sig*1e6:.3f})\\times10^{{-6}}\\,^\\circ\\mathrm{{C}}^{{-1}}$.\n\n")
    tex += r'''\subsection*{Goodness of fit}
\[
\chi^2=\sum_i\frac{(y_i-\widehat y_i)^2}{\sigma_{y_i}^2},
\qquad \chi^2_\nu=\frac{\chi^2}{N-p}
\]
The above conventional expression applies to independent errors in $y$ only.
Our primary two-axis fit minimizes squared normalized corrections in both
$x$ and $y$, and reports the corresponding total $\chi^2$ divided by $N-p$.
$R^2=1-\mathrm{SSE}/\mathrm{SST}$ is computed from vertical residuals at
recorded x-values. Exact reference-angle and thermal-lag systematics are not
captured by these reported statistical uncertainties.
'''
    # Course handout regression formula, demonstrated separately from primary EIV.
    ols0=g0['ols']
    xo=xg
    yo=green[:,1]
    count=len(xo)
    Delta=count*np.sum(xo**2)-np.sum(xo)**2
    yhat5=float(linear(np.array([xo[0]]),ols0['p'])[0])
    residual5=float(yo[0]-yhat5)
    tex += r"""\subsection*{Course handout: OLS slope, uncertainty, and residual}
\[
m=\frac{N\sum_i x_i y_i-(\sum_i x_i)(\sum_i y_i)}{N\sum_i x_i^2-(\sum_i x_i)^2},\qquad
s_m=\sqrt{\frac{Ns_{y,x}^2}{\Delta}},\qquad
s_{y,x}^2=\frac{\sum_i(y_i-mx_i-b)^2}{N-2}.
\]
"""
    tex += (f"For green trial 1, $N={count}$ and $\\Delta={Delta:.0f}$. The ordinary least-squares "
            f"slope is $m={ols0['p'][0]:.4f}\\pm{ols0['s'][0]:.4f}$ fringes/$\\mu$m.\n"
            f"At $\\Delta x={xo[0]:.1f}\\,\\mu$m, the observed fringe count is ${yo[0]:.0f}$, "
            f"the prediction is ${yhat5:.3f}$, and residual is ${residual5:.3f}$ fringes.\n"
            r"The effective vertical residual-bar uncertainty in a linear fit is "
            r"$\sigma_{r,\mathrm{eff}}\simeq\sqrt{\sigma_N^2+(m\sigma_{x,\mathrm{ind}})^2}$."
            "\n\n")
    (out_dir/'appendix_calculations.tex').write_text(tex,encoding='utf-8')

    report=[]
    report.append('INTERFEROMETRY FIT RESULTS AND IMPORTANT INTERPRETATION NOTES')
    report.append('='*68)
    report.append(f'Green wavelength per trial: {wave_vals.round(3).tolist()} nm')
    report.append(f'Green independent-run mean = {wave_mean:.3f} nm')
    report.append(f'Trial-to-trial standard error of mean = {wave_sem:.3f} nm (3 trials)')
    report.append(f'Propagated mean fit uncertainty (independent formal fit errors) = {wave_formal_mean:.3f} nm')
    report.append(f'WORKING downstream wavelength uncertainty = {wave_work_sigma:.3f} nm, the larger of those two')
    report.append('This working choice is a transparent conservative convention, not a unique exact 68% CI.')
    report.append('Common instrument calibration/systematic effects are not estimated by repeated measurements.')
    report.append('')
    for row in rows:
        report.append(f"{row['dataset']}: slope={row['slope']:.6g} +/-{row['sigma_slope']:.4g}; "
            f"b={row['intercept']:.5g} +/-{row['sigma_intercept']:.3g}; "
            f"R²={row['R_squared']:.5f}; χ²={row['chi_squared']:.3f}; "
            f"DOF={row['degrees_freedom']}; reduced χ²={row['reduced_chi_squared']:.3f}")
    report.append('')
    report.append(f'Prism nominal comparison: 134°+ n={rows[3]["refractive_index"]:.5f}; '
                  f'134°- n={rows[4]["refractive_index"]:.5f}.')
    report.append(f'Prism 134+ OLS free-intercept diagnostic: slope={prism_fits[0]["free_ols"]["p"][0]:.2f}, b={prism_fits[0]["free_ols"]["p"][1]:.2f} fringes')
    report.append(f'Prism 134- OLS free-intercept diagnostic: slope={prism_fits[1]["free_ols"]["p"][0]:.2f}, b={prism_fits[1]["free_ols"]["p"][1]:.2f} fringes')
    report.append(f'Simple joint directional-asymmetry diagnostic: reference offset ≈ {ref_shift_deg:.3f} degrees')
    report.append(f'Joint model slope={m_joint:.2f} rad^-2, n={joint_n:.4f}.')
    report.append('The joint offset is a MODEL DIAGNOSTIC, not a directly measured correction.')
    report.append('Possible assumed reference-angle bias versus inferred combined refractive index:')
    for degree, mm, nn in sensitivity:
        report.append(f'  assumed shift {degree:+.2f}° -> slope {mm:.2f}, n {nn:.4f}')
    report.append('')
    report.append(f'Aluminium alpha = ({alpha:.6g} +/- {alpha_sig:.2g}) 1/°C')
    report.append('Two measurements at ΔT=16.9°C have N=119 and N=122. Both retained.')
    report.append('The point (0°C,0 fringe) is a defined baseline and excluded from fitted data.')
    report.append('')
    report.append('UNCERTAINTY CONVENTIONS')
    report.append('Resolution half-widths 0.5 µm, 0.5°, 0.05°C, 0.005 mm are treated as')
    report.append('uniform rounding bounds and divided by sqrt(3) for 1-sigma propagation.')
    report.append('0.5 fringe is separately treated as an estimated 1-sigma count uncertainty.')
    report.append('x error bars include shared starting reading where appropriate; the fit')
    report.append('uses only independent x uncertainties because free intercepts absorb')
    report.append('a common additive baseline shift. Prism reference-angle systematic')
    report.append('is assessed separately and is NOT in formal n uncertainty.')
    report.append('Residual plots show ordinary vertical residuals; vertical error bars show')
    report.append('sqrt(sigma_N^2 + (dy/dx * sigma_x,independent)^2).')
    report.append('The chi-square of the two-axis fit measures both normalized x and y')
    report.append('corrections, NOT just the vertical residuals seen on the bottom panel.')
    report.append('')
    report.append('LIMITATIONS / REPORT DISCUSSION')
    report.append('- No artificially added origin for green laser or prism.')
    report.append('- Only five green data points per run, fewer than manual recommendation.')
    report.append('- Missed fringes can occur randomly between trials but create cumulative,')
    report.append('  correlated step offsets within each trial; residuals may suggest them')
    report.append('  but cannot prove one was missed, or exactly how many.')
    report.append('- Mechanical jitter is a likely count-tracking problem, not additional')
    report.append('  measured micrometer-position uncertainty. Backlash not claimed when')
    report.append('  screw operated continuously in one direction.')
    report.append('- 134° prism reference selected once; unknown true perpendicular offset')
    report.append('  is not fully described by +/-0.5° scale-reading resolution alone.')
    report.append('- Aluminium temperature gradient / thermometer lag could explain')
    report.append('  different fringe counts at same recorded temperature. Not quantified.')
    report.append('- A few points may be missed whole fringes; do not discard them simply')
    report.append('  to improve fit; document any justified exclusion or alternative fit.')
    report.append('- Laser wavelength is shared across subsequent experiments, so its')
    report.append('  contribution cannot be treated as a new independent draw per point.')
    report.append('- The measured red LED wavelength, coherence length, and bandwidth')
    report.append('  cannot be computed without red LED observations.')
    report.append('- A small reduced chi² / R² mismatch is not automatically evidence of')
    report.append('  a bad plot; inspect assumptions, reference offsets and residual patterns.')
    report.append('')
    report.append('COURSE HANDOUT CROSS-CHECK')
    report.append('Each results row includes ordinary least squares slope and its standard')
    report.append('error obtained from residual variance (the class handout formulas).')
    report.append('Primary estimated slopes above account for both x and y uncertainties;')
    report.append('their parameter uncertainties do not rescale input errors to force χ²ν=1.')
    (out_dir/'analysis_notes.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report[:18]))
    print(f'\nResults written to {out_dir.resolve()}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,default=Path('interferometry_results'))
    args=parser.parse_args()
    main(args.data_dir,args.out)
