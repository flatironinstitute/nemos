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

from nemos.third_party.jaxopt.jaxopt._src.tree_util import broadcast_pytrees
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_map
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_reduce
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_add
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_sub
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_mul
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_scalar_mul
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_add_scalar_mul
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_dot
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_vdot
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_vdot_real
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_div
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_sum
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_l2_norm
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_where
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_zeros_like
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_ones_like
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_negative
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_inf_norm
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_conj
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_real
from nemos.third_party.jaxopt.jaxopt._src.tree_util import tree_imag