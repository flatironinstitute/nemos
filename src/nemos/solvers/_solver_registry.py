from . import (
    JaxoptGradientDescent,
    JaxoptProximalGradient,
    JaxoptBFGS,
    JaxoptLBFGS,
    WrappedProxSVRG,
    WrappedSVRG,
)

solver_registry = {
    "GradientDescent": JaxoptGradientDescent,  # jaxopt.GradientDescent,
    "ProximalGradient": JaxoptProximalGradient,
    "LBFGS": JaxoptLBFGS,
    "BFGS": JaxoptBFGS,
    "SVRG": WrappedSVRG,
    "ProxSVRG": WrappedProxSVRG,
}
