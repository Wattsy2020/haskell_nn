-- Goal: Use the modules to train and do inference with a machine learning model
import Autograd
  ( Node (Variable),
    derivative,
    evalInputs,
    evalNode,
    showDerivative,
  )
import Data.Map qualified as Map
import NN (sigmoid)

main = do
  let var = Variable "x"
  let expr = 4 * (var + 5) * var
  print expr
  putStrLn $ showDerivative expr
  let varMap = Map.singleton "x" 5
  print $ evalNode expr varMap
  print $ evalNode (derivative expr) varMap

  -- plot a parabola
  let parabola = var * var
  let xs = [-4 .. 4]
  let ys = evalInputs parabola "x" xs
  let dydxs = evalInputs (derivative parabola) "x" xs
  print (parabola, showDerivative parabola)
  print $ zip3 xs ys dydxs

  -- check double negatives are handled well
  let expr1 = negate (negate (5 * var)) - (negate var) + (negate (negate 2 * var))
  print expr1
  putStrLn $ showDerivative expr1

  -- check that multiplication can be simplified
  let expr2 = (negate 2 * (var ** 2) * 2) + (5 * var ** 2) + (var ** 2) + ((4 * var ** 3) - (var ** 3))
  print expr2
  putStrLn $ showDerivative expr2

  -- differentiate a fairly complicated function
  let expr3 = var * var / (1 + var)
  print expr3
  putStrLn $ showDerivative expr3

  -- differentiate floating function
  let expr4 = exp (cos (var ** 2))
  print expr4
  putStrLn $ showDerivative expr4
  print (evalNode expr4 (Map.singleton "x" $ sqrt pi), exp (-1))

  -- test out neural network functions
  let sigmoidExpr = sigmoid var
  print sigmoidExpr
  print $ evalNode sigmoidExpr (Map.singleton "x" 5)
  putStrLn $ showDerivative sigmoidExpr
