# Tier 1 — Arithmetic

## Question
What is 17 × 23?

## Approach
Multiply 17 by 23 using Python's arbitrary-precision integers, and
cross-check via the distributive property.

## Code
```python
direct = 17 * 23
factored = (17 * 20) + (17 * 3)
print(f"17 * 23 = {direct}")
print(f"(17 * 20) + (17 * 3) = {factored}")
assert direct == factored
```

## Output
```
17 * 23 = 391
(17 * 20) + (17 * 3) = 391
```

## Re-run anytime
Save the code block to a .py file and run it. No LLM, no tokens.