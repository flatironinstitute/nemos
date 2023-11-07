# Proximal Methods in Optimization

## Introduction 

In optimization theory, the proximal operator is a mathematical tool used to solve non-differentiable optimization
problems or to simplify complex ones.

The proximal operator of a function $ f: \mathbb{R}^n \rightarrow \mathbb{R} \cup \{+\infty\} $ is defined as follows:

$$
\text{prox}_f(v) = \arg\min_x \left( f(x) + \frac{1}{2}\Vert x - v\Vert_2 ^2 \right)
$$

Here $ \text{prox}_f(v) $ is the value of $ x $ that minimizes the sum of the function $ f(x) $ and the
squared Euclidean distance between $ x $ and some point $ v $. The parameter $ f $ typically represents
a regularization term or a penalty in the optimization problem, and $ v $ is typically a vector
in the domain of $ f $.

The proximal operator can be thought of as a generalization of the projection operator. When $ f $ is the
indicator function of a convex set $ C $, then $ \text{prox}_f $ is the projection onto $ C $, since
it finds the point in $ C $ closest to $ v $.

Proximal operators are central to the implementation of proximal gradient[^1] methods and algorithms like where they
help to break down complex optimization problems into simpler sub-problems that can be solved iteratively.

## Proximal Operators in Proximal Gradient Algorithms

Proximal gradient algorithms are designed to solve optimization problems of the form:

$$
\min_{x \in \mathbb{R}^n} g(x) + f(x)
$$

where $ g $ is a differentiable (and typically convex) function, and $ f $ is a (possibly non-differentiable) convex
function that imposes certain structure or sparsity in the solution. The proximal gradient method updates the
solution iteratively through a two-step process:

1. **Gradient Step on $ g $**: Take a step towards the direction of the negative gradient of $ g $ at the current
estimate $ x_k $, with a step size $ \alpha_k $, leading to an intermediate estimate $ y_k $:
   $$
   y_k = x_k - \alpha_k \nabla g(x_k)
   $$
2. **Proximal Step on $ f $**: Apply the proximal operator of $ f $ to the intermediate
estimate $ y_k $ to obtain the new estimate $ x_{k+1} $:

   $$
   x_{k+1} = \text{prox}_{ f}(y_k) = \arg\min_x \left( f(x) + \frac{1}{2\alpha_k}\Vert x - y_k \Vert_2 ^2 \right)
   $$

The gradient step aims to reduce the value of the smooth part of the objective $ g $, and the proximal step
takes care of the non-smooth part $ f $, often enforcing properties like sparsity due to regularization terms
such as the $ \ell_1 $ norm.

By iteratively performing these two steps, the proximal gradient algorithm converges to a solution that
balances minimizing the differentiable part $ g $ while respecting the structure imposed by the non-differentiable
part $ f $. The proximal operator effectively "proximates" the solution at each iteration,
taking into account the influence of the non-smooth term $ f $, which would be otherwise challenging to
handle due to its potential non-differentiability.

[^1]: Parikh, Neal, and Stephen Boyd. "Proximal Algorithms, ser. Foundations and Trends (r) in Optimization." (2013).
