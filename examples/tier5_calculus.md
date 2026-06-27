# Tier 5 — Calculus

## Question
What is the integral of x² from 0 to 1?

## Approach
Use sympy for the exact symbolic answer, scipy for a numerical
cross-check, and matplotlib to plot the integrand and the area.

## Code
```python
import sympy as sp
from scipy import integrate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

x = sp.symbols("x")
exact = sp.integrate(x**2, (x, 0, 1))
print("Symbolic:", exact)

f = lambda t: t**2
approx, err = integrate.quad(f, 0, 1)
print(f"Numerical: {approx:.10f}  (error ~ {err:.2e})")

xs = np.linspace(0, 1, 200)
plt.fill_between(xs, xs**2, alpha=0.3, label="area")
plt.plot(xs, xs**2, label="y = x²")
plt.title(f"∫ x² dx from 0 to 1 = {exact}")
plt.xlabel("x"); plt.ylabel("y")
plt.legend()
plt.savefig("integral.png", dpi=100, bbox_inches="tight")
```

## Output
```
Symbolic: 1/3
Numerical: 0.3333333333  (error ~ 3.70e-15)
```

## Mermaid
```mermaid
graph LR
  A["Integrand x²"] --> B["Antiderivative x³/3"]
  B --> C["Evaluate 0 to 1"]
  C --> D["Result = 1/3"]
```

## Re-run anytime
Save the code blocks above to a .py file and run it. The Mermaid
diagram can be rendered to SVG with `mmdc -i diagram.mmd -o out.svg`.