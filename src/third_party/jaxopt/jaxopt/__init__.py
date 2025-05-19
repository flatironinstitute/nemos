# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

from third_party.jaxopt.jaxopt import implicit_diff
from third_party.jaxopt.jaxopt import isotonic
from third_party.jaxopt.jaxopt import loss
from third_party.jaxopt.jaxopt import objective
from third_party.jaxopt.jaxopt import projection
from third_party.jaxopt.jaxopt import prox

from third_party.jaxopt.jaxopt._src.anderson import AndersonAcceleration
from third_party.jaxopt.jaxopt._src.anderson_wrapper import AndersonWrapper
from third_party.jaxopt.jaxopt._src.armijo_sgd import ArmijoSGD
from third_party.jaxopt.jaxopt._src.base import OptStep
from third_party.jaxopt.jaxopt._src.backtracking_linesearch import BacktrackingLineSearch
from third_party.jaxopt.jaxopt._src.bfgs import BFGS
from third_party.jaxopt.jaxopt._src.bisection import Bisection
from third_party.jaxopt.jaxopt._src.block_cd import BlockCoordinateDescent
from third_party.jaxopt.jaxopt._src.broyden import Broyden
from third_party.jaxopt.jaxopt._src.cd_qp import BoxCDQP
from third_party.jaxopt.jaxopt._src.cvxpy_wrapper import CvxpyQP
from third_party.jaxopt.jaxopt._src.eq_qp import EqualityConstrainedQP
from third_party.jaxopt.jaxopt._src.fixed_point_iteration import FixedPointIteration
from third_party.jaxopt.jaxopt._src.gauss_newton import GaussNewton
from third_party.jaxopt.jaxopt._src.gradient_descent import GradientDescent
from third_party.jaxopt.jaxopt._src.hager_zhang_linesearch import HagerZhangLineSearch
from third_party.jaxopt.jaxopt._src.iterative_refinement import IterativeRefinement
from third_party.jaxopt.jaxopt._src.lbfgs import LBFGS
from third_party.jaxopt.jaxopt._src.lbfgsb import LBFGSB
from third_party.jaxopt.jaxopt._src.levenberg_marquardt import LevenbergMarquardt
from third_party.jaxopt.jaxopt._src.mirror_descent import MirrorDescent
from third_party.jaxopt.jaxopt._src.nonlinear_cg import NonlinearCG
from third_party.jaxopt.jaxopt._src.optax_wrapper import OptaxSolver
from third_party.jaxopt.jaxopt._src.osqp import BoxOSQP
from third_party.jaxopt.jaxopt._src.osqp import OSQP
from third_party.jaxopt.jaxopt._src.polyak_sgd import PolyakSGD
from third_party.jaxopt.jaxopt._src.projected_gradient import ProjectedGradient
from third_party.jaxopt.jaxopt._src.proximal_gradient import ProximalGradient
from third_party.jaxopt.jaxopt._src.scipy_wrappers import ScipyBoundedLeastSquares
from third_party.jaxopt.jaxopt._src.scipy_wrappers import ScipyBoundedMinimize
from third_party.jaxopt.jaxopt._src.scipy_wrappers import ScipyLeastSquares
from third_party.jaxopt.jaxopt._src.scipy_wrappers import ScipyMinimize
from third_party.jaxopt.jaxopt._src.scipy_wrappers import ScipyRootFinding
from third_party.jaxopt.jaxopt._src.zoom_linesearch import ZoomLineSearch

warnings.warn(
    "JAXopt is no longer maintained. See https://docs.jax.dev/en/latest/ for"
    " alternatives.",
    DeprecationWarning,
)
