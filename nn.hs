module NN (sigmoid) where

import Autograd (Node (Variable))

sigmoid :: (Eq a, Floating a) => Node a -> Node a
sigmoid x = 1 / (1 + exp (-x))