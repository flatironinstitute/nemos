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

from nemos.third_party.jaxopt.jaxopt.jaxopt import implicit_diff
from nemos.third_party.jaxopt.jaxopt.jaxopt import loss
from nemos.third_party.jaxopt.jaxopt.jaxopt import objective
from nemos.third_party.jaxopt.jaxopt.jaxopt import projection
from nemos.third_party.jaxopt.jaxopt.jaxopt import prox

from nemos.third_party.jaxopt.jaxopt.jaxopt._src.armijo_sgd import ArmijoSGD
from nemos.third_party.jaxopt.jaxopt.jaxopt._src.base import OptStep
from nemos.third_party.jaxopt.jaxopt.jaxopt._src.backtracking_linesearch import (
    BacktrackingLineSearch,
)
from nemos.third_party.jaxopt.jaxopt.jaxopt._src.bfgs import BFGS
from nemos.third_party.jaxopt.jaxopt.jaxopt._src.gradient_descent import GradientDescent
from nemos.third_party.jaxopt.jaxopt.jaxopt._src.hager_zhang_linesearch import (
    HagerZhangLineSearch,
)
from nemos.third_party.jaxopt.jaxopt.jaxopt._src.lbfgs import LBFGS
from nemos.third_party.jaxopt.jaxopt.jaxopt._src.proximal_gradient import ProximalGradient
from nemos.third_party.jaxopt.jaxopt.jaxopt._src.zoom_linesearch import ZoomLineSearch

warnings.warn(
    "JAXopt is no longer maintained. See https://docs.jax.dev/en/latest/ for"
    " alternatives.",
    DeprecationWarning,
)
