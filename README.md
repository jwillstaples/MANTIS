MANTIS

Helpful Links:

FreeCodeCamp implementing AlphaZero from scratch:
https://www.youtube.com/watch?v=wuSQpLinRB4

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

* Passing 1 board, 2000 times takes 16.74 seconds (120 evals/sec)
* Passing 1 board, 20000 times takes 180 seconds (110 evals/sec)

Connect Four:
Pre-tree passing: 40 sec/game
Tree Passing: 