MANTIS

We're creating the first 2v2 chess (Bughouse) engine, by developing a reinforcement learning framework, iterated on existing ML architectures.

Helpful Links:

FreeCodeCamp implementing AlphaZero from scratch:
https://www.youtube.com/watch?v=wuSQpLinRB4

## VM shit
* https://ssh.cloud.google.com/v2/ssh/projects/mantis-415520/zones/us-central1-a/instances/instance-20240319-165546?authuser=0&hl=en_US&projectNumber=1032060796269&useAdminProxy=true
* 10.128.0.3
* 34.72.28.14
* nohup python3 your_script.py &
* tail -f nohup.out

## Visioning Presentation

P1 - Ashrit

P2 - Andy

V1 - Will

V2 - Elliot

## Efficiency Improvement

Neural Network Input Efficiency:

* Passing 10 boards at a time, 200 times takes 2.5 seconds (800 evals/sec)
* Passing 100 boards at a time, 200 times takes 5.6 seconds (3500 evals/sec)
* Passing 1000 boards at a time, 200 times takes 37.27 seconds (5366 evals/sec)
* Passing 5000 boards at a time, 200 times takes 187 seconds (5347 evals/sec)
* Passing 1 board, 2000 times takes 16.74 seconds (120 evals/sec)
* Passing 1 board, 20000 times takes 180 seconds (110 evals/sec)

Connect Four:

(30 games)

Pre-tree passing: 40 sec/game
Tree Passing: 8 sec/game
95% of the time is GPU forward calls

Parallel, no tree passing: 9 sec / game
Parallel, tree passing: 7 sec / game
Parallel in Parallel, tree passing 3 sec / game
45% of the time is GPU forward calls

(300 games)

Pre-tree passing: 40 sec/game
Parallel, tree passing: 5 sec / game 
Parallel in Parallel, tree passing: 2.5 sec / game

## Checkpoints

#### 1
* Andy
* Will
* Ashrit

#### 2
* Andy
* Elliot
* Ashrit

#### 3
* Elliot
* Will



Looking at the training, parallel and serial, there is a position with a mate in 1, the p_vec does not return 1, instead looking at the value score is not exactly 1 or -1.
