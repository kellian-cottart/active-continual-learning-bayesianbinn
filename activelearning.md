# Dynamic Threshold Computation for Active Learning

We define the following variables:

- $\tau$ : the active learning threshold
- $K$ : number of samples for quantization
- $B$ : target budget (fraction of samples we want to label)
- $Q_{\text{rate}}$ : current query rate (fraction of samples already labeled)
- $\epsilon$ : signed error between current query rate and target budget
- $\gamma$ : exponent controlling the adaptation speed


1. Compute the error between current query rate and target budget:

$$
\epsilon = Q_{\text{rate}} - B
$$

2. Continuous threshold before quantization

$$
\tau_{\text{cont}} = B + \text{sign}(\epsilon) \cdot |\epsilon|^{\gamma}
$$

- If $Q_{rate} > B$ ($\epsilon > 0$), the threshold increases faster → more selective, fewer queries.  
- If $Q_{rate} < B$ ($\epsilon < 0$), the threshold decreases proportionally → less selective, more queries.

3. Quantize the threshold according to the number of samples

$$
n = \text{round}(K \cdot \tau_{\text{cont}})
$$

$$
n = \min(\max(n, 0), K)
$$


4. Final threshold

$$
\tau = \frac{n}{K}
$$  

This ensures the threshold $\tau$ lies in the quantized set:

$$
\tau \in \left\{ 0, \frac{1}{K}, \frac{2}{K}, \dots, \frac{K}{K} \right\}
$$

## Summary

- The threshold dynamically adapts to the difference between current query rate and target budget.  
- Overshooting the budget increases the threshold (more selective), undershooting decreases it (more permissive).  
- Quantization ensures the threshold matches discrete sample selection steps.
