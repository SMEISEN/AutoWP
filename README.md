# AutoWP: Automated wind power forecasts with limited computing resources using an ensemble of diverse wind power curves

AutoWP addresses two challenges:
- Achieving good accuracy in Wind Power (WP) forecasting with low computational effort
- Handling regular or irregular interventions in the WP generation capabilities

## Methodology

The underlying idea of AutoWP is to represent a new WP turbine as a convex linear combination of WP curves from a sufficiently diverse ensemble. The method consists of three steps: i) create the ensemble of normalized WP curves, ii) form the normalized ensemble WP curve by the optimally weighted sum of the WP curves in the ensemble, and iii) re-scale the ensemble WP curve with the new WP turbine’s peak power rating.

![autowp_pipeline](https://github.com/SMEISEN/AutoWP/assets/33990691/46b4f23c-4a8f-423e-8e20-a24e2b02ff5d)

## Installation

To install this project, perform the following steps.
1) Clone the project
2) Open a terminal of the virtual environment where you want to use the project
3) cd AuroWP
4) pip install . or pip install -e . if you want to install the project editable.

## How to use

Exemplary evaluations using AutoWP are given in the examples folder.

### Model pool

The model pool is based on a selection of 10 WP curves from the [windpowerlib](https://github.com/wind-python/windpowerlib). The selection reduces redundancy and preserves diversity. Since these WP curves provided by turbine Original Equipment Manufacturers (OEMs) have the hub height as reference height, height correction of the wind speed forecast based on the wind profile power law is used. 

### Evaluation types

The offline evaluation optimizes the ensemble weights using the entire training data and does not adapt the weights over time.

## Citation

If you use this method please cite the corresponding paper:
> Stefan Meisenbacher et al. 2024. AutoWP: Automated wind power forecasts with limited computing resources using an ensemble of diverse wind power curves. In preparation.

## Funding

This project is funded by the Helmholtz Association under the Program “Energy System Design” and the Helmholtz Association?s Initiative and Networking Fund through Helmholtz AI.

## References

The example data is from the wind power forecasting track of the Global Energy Forecasting Competition (GEFCom) 2014:
> T. Hong, P. Pinson, S. Fan, H. Zareipour, A. Troccoli, and R. J. Hyndman, “Probabilistic energy forecasting: Global energy forecasting competition 2014 and beyond”, International Journal of Forecasting, vol. 32, no. 3, pp. 896–913, 2016.
