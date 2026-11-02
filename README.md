# Interferometry lab: reproducible analysis

This package uses the three original uploaded CSVs without modifying them and produces publication-style Matplotlib plots, fit statistics, a table of physical results, and a LaTeX appendix.

## Run

Install the dependencies with:

```bash
python -m pip install numpy scipy matplotlib
```

From the root of this package, run:

```bash
python interferometry_analysis.py --data-dir . --out results
```

Outputs include `results/green_trial_1.pdf` (and trials 2 and 3), `results/green_comparison.pdf`, `results/prism_134_plus.pdf`, `results/prism_134_minus.pdf`, `results/prism_comparison.pdf`, and `results/aluminium_expansion.pdf`, plus corresponding PNGs. See `results/fit_results.csv`, `results/analysis_notes.txt`, `results/appendix_calculations.tex` and `appendix_preview.pdf`.

To use the sample appendix calculations in Overleaf, insert `\input{results/appendix_calculations.tex}` in a document with `amsmath` loaded, or copy the portions needed.

## Fit convention (important)

- Green laser: free intercept, **only the five observed points in each trial**.
- Prism: zero-intercept theoretical model, fitted separately in each direction. A separate free-intercept diagnostic and joint approximate reference-angle-offset diagnostic are included in the notes. Do not describe the nominal results as conclusive in view of the disagreement between directions.
- Aluminium: free intercept, with the chosen baseline at `(0,0)` drawn as an unfilled reference point but **excluded from the fit**. The last two observed counts at identical recorded temperature are retained.
- The primary reported fits account for independent uncertainties in both x and y (errors-in-variables via a numerical least-squares fit). The table also includes the standard OLS slopes and scatter-based standard errors from the class linear-fitting handout. They are not the same uncertainty convention.
- Maximal reading half-widths ±0.5 µm, ±0.5°, ±0.05 °C, ±0.005 mm are converted to 1-sigma standard uncertainties by dividing by √3, using a uniform rounding-error model. This convention is *an assumption* and can be changed in the script if the TA treats the stated values as 1-sigma.
- ±0.5 fringe is treated as an estimated 1-sigma fringe-identification uncertainty. This does **not** quantify lost complete fringes from fast motion.
- Shared initial micrometer and thermometer readings affect every displacement/temperature-difference point. The independent part is used in the free-intercept fit, the total difference uncertainty is used in the displayed horizontal error bars. Prism reference-angle bias is treated separately as an unquantified model systematic.
- Vertical residual error bars display an approximate propagated **measurement** uncertainty, not a confidence interval for the fitted line. Fit parameter covariance is not added independently to each residual.
- Reduced chi-square is calculated using the **two-axis objective function**, divided by `n - number_of_physical_fit_parameters`. The displayed residuals are conventional vertical residuals at measured x. This distinction is deliberate.
- Fitted 1-sigma parameter uncertainties are based on the specified input standard uncertainties, using a local Jacobian covariance approximation. The model and assumptions should be checked where reduced chi-square is large.
- Wavelength used downstream is the average of three fit-derived green wavelengths. The supplied working uncertainty is the larger of the trial-to-trial standard error and the independent-fit uncertainty of the mean. This is a **conservative reporting convention**, not a uniquely defined statistical confidence interval; both components are printed explicitly in the notes.
- Red LED wavelength/coherence length/spectral bandwidth are not included because the uploaded datasets do not contain red LED observations.

## Lab report caveats

Only five measurements exist per green run although the manual suggests 10–20. Missing whole fringes can occur randomly during an experiment while producing *cumulative, correlated* counting shifts later in a run. A residual pattern can suggest a miscount but is not proof. Backlash is not presumed when the screw was advanced in a single direction. Thermal gradients/lag, the precise perpendicular reference of the prism, and shared calibration effects cannot be fully quantified from the existing data.

If your TA requires the strict OLS procedures from the provided class handout rather than an errors-in-variables estimate, use the OLS columns and state that the x-error bars are shown but x-uncertainty is not included in the OLS fit. Do not mix the OLS slopes with the parameter errors of the other fitting model.
