Simple FC NN in Excel:

![simple neural network](./neural%20network.png)


# Formula

$$h1 = w1 * i1 + w2 * i2$$

$$h2 = w3 * i1 + w4 * i2$$

$$a_h1 = \sigma(h1)$$

$$a_h2 = \sigma(h2)$$

$$o1 = w5 * a_h1 + w6 * a_h2$$

$$o2 = w7 * a_h1 + w8 * a_h2$$

$$a_o1 = \sigma(o1)$$

$$a_o2 = \sigma(o2)$$

$$E_{total} = E1+E2 $$

$$E1 = 0.5 * (t1-a_o1) $$

$$E2 = 0.5 * (t2-a_o2) $$

$$∂E_{total}/∂w5 = ∂(E1 + E2)/∂w5$$

$$∂E_{total}/∂w5 = ∂E1/∂w5$$ 
E2=0 (i.e) independent

$$∂E_{total}/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5$$

$$∂E1/∂a_o1 =  ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1)$$

$$∂a_o1/∂o1 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)$$		

$$∂o1/∂w5 = a_h1$$

$III^{ly}$

$$∂E_{total}/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1$$				

$$∂E_{total}/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2$$				

$$∂E_{total}/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1$$				

$$∂E_{total}/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2$$


$$∂E1/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5$$								

$$∂E2/∂a_h1 = (a_02 - t2) * a_o2 * (1 - a_o2) * w7$$

$$∂E_{total}/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7$$							

$$∂E_{total}/∂a_h2 = (a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8$$


$$∂E_{total}/∂w1 = ∂E_{total}/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1$$

$$∂E_{total}/∂w2 = ∂E_{total}/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2$$				

$$∂E_{total}/∂w3 = ∂E_{total}/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3$$


$$∂E_{total}/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1$$											

$$∂E_{total}/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2$$											

$$∂E_{total}/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1$$											

$$∂E_{total}/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2$$


## Playing with different learning_rate:

![eta=0.1](./different%20lr/eta_0.1.png)

![eta=0.2](./different%20lr/eta_0.2.png)

![eta=0.5](./different%20lr/eta_0.5.png)

![eta=0.8](./different%20lr/eta_0.8.png)

![eta=1](./different%20lr/eta_1.png)

![eta=2](./different%20lr/eta_2.png)