# Linear Quadratic Regulator

## background

### optimality condition

- KKT conditions은 Nonconvex problem의 경우, optimality를 위한 필요조건이고, convex problem의 경우, optimality를 위한 필요충분조건이다.
- Lagrangian function : $L(x, \mu, \lambda) := f(x) + \mu^Tg(x) + \lambda^Th(x)$
- KKT condition :
    1. $$\left. \frac{\partial L(x, \mu, \lambda)}{\partial x} \right|_{x=x^*} = 0$$

    2. $$g_i(x^*)\leq 0, \forall i,\;\; h_j(x^*) = 0, \forall j$$

    3. $$\mu_i \geq 0, \forall i$$

    4. $$\sum_{i} \mu_i g_i(x^*) = 0$$

### Hamilton 

## discrete time, finite time horizon

