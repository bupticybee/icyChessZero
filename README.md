# Icy Elephant
Like Go, Chinese chess is an ancient chess game, After I trained a CNN model to play Go ( https://github.com/bupticybee/icygo ), I wonder if the similar approach can be used to play chinese chess. 

So I trained a CNN to do the same thing in chinese chess. I played with it, Althrough the network preduces some interesting result, it is not strong enough, in the 10 games I played with the Neural network, I win all of them by large margin.

Here are the first five move in a CNN self-play:

![](./img/play1.PNG)
![](./img/play2.PNG)
![](./img/play3.PNG)
![](./img/play4.PNG)
![](./img/play5.PNG)


The data I use can be downloaded here:
  https://pan.baidu.com/s/1JCmweZUREJxjIMXQlL-o9g

Follow the code in  chess_policy_resnet10.ipynb to train the model.
After you trained your model, follow the code in  play_against_computer.ipynb to play with the model you train.

Have Fun