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

from third_party.jaxopt.jaxopt._src.tree_util import broadcast_pytrees
from third_party.jaxopt.jaxopt._src.tree_util import tree_map
from third_party.jaxopt.jaxopt._src.tree_util import tree_reduce
from third_party.jaxopt.jaxopt._src.tree_util import tree_add
from third_party.jaxopt.jaxopt._src.tree_util import tree_sub
from third_party.jaxopt.jaxopt._src.tree_util import tree_mul
from third_party.jaxopt.jaxopt._src.tree_util import tree_scalar_mul
from third_party.jaxopt.jaxopt._src.tree_util import tree_add_scalar_mul
from third_party.jaxopt.jaxopt._src.tree_util import tree_dot
from third_party.jaxopt.jaxopt._src.tree_util import tree_vdot
from third_party.jaxopt.jaxopt._src.tree_util import tree_vdot_real
from third_party.jaxopt.jaxopt._src.tree_util import tree_div
from third_party.jaxopt.jaxopt._src.tree_util import tree_sum
from third_party.jaxopt.jaxopt._src.tree_util import tree_l2_norm
from third_party.jaxopt.jaxopt._src.tree_util import tree_where
from third_party.jaxopt.jaxopt._src.tree_util import tree_zeros_like
from third_party.jaxopt.jaxopt._src.tree_util import tree_ones_like
from third_party.jaxopt.jaxopt._src.tree_util import tree_negative
from third_party.jaxopt.jaxopt._src.tree_util import tree_inf_norm
from third_party.jaxopt.jaxopt._src.tree_util import tree_conj
from third_party.jaxopt.jaxopt._src.tree_util import tree_real
from third_party.jaxopt.jaxopt._src.tree_util import tree_imag