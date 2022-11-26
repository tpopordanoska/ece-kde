# Calibration Error Estimator
This is the official code repository for ["A Consistent and Differentiable Lp Canonical Calibration Error Estimator"](
https://arxiv.org/abs/2210.07810), published in NeurIPS 2022.

The paper proposes $ECE^{KDE}$, a consistent and differentiable estimator of the Lp calibration error. To model a density
over a simplex, we use Kernel Density Estimation (KDE) with Dirichlet kernels. This estimator can tractably capture the
highest form of calibration, called canonical (or distribution) calibration, which requires the entire probability 
vector to be calibrated.

## Usage
$ECE^{KDE}$ can be directly optimized alongside any loss function in a calibration regularized training objective:

$$f = \arg\min_{f\in \mathcal{F}}\, \Bigl(\operatorname{Risk}(f) + \lambda \cdot \operatorname{CE}(f)\Bigr). $$

The weight $\lambda$ is chosen via cross-validation. 

Additionally, the estimator can be used as a metric to evaluate canonical (distribution), marginal (classwise) and top-label (confidence) calibration. 


## To use it in your project
Copy the file `ece_kde.py` to your repo. You can obtain the estimate of CE with the method `get_ece_kde`. The bandwidth 
of the kernel can either be manually set, or chosen by maximizing the leave-one-out likelihood with the method 
`get_bandwidth`. 
For example, an estimate of canonical CE as defined in Equation 9 in the paper can be obtained with:
```
# Generate dummy probability scores and labels
f = torch.rand((50, 3))
f = f / torch.sum(f, dim=1).unsqueeze(-1)
y = torch.randint(0, 3, (50,))

get_ece_kde(f, y, bandwidth=0.001, p=1, mc_type='canonical', device='cpu')
```
The code is still in its preliminary version. A demo will be available soon.

## Reference
If you found this work or code useful, please cite:

```
@inproceedings{Popordanoska2022b,
  title={A Consistent and Differentiable $L_p$ Canonical Calibration Error Estimator},
  AUTHOR = {Popordanoska, Teodora and Sayer, Raphael and Blaschko, Matthew B.},
  YEAR = {2022},
  booktitle = {Advances in Neural Information Processing Systems},
}
```

## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).
