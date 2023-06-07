-- Find roots in functions using Newton's method
import Autograd (Node (..), derivative, evalNode)
import Data.Map qualified as Map

solve :: (Eq a, Floating a) => Node a -> a -> a
solve node initGuess = initGuess - evalNode (node / derivative node) (Map.singleton "x" initGuess)

solveIterate :: (Eq a, Floating a) => Node a -> a -> [a]
solveIterate node = iterate (solve node)

main = do
  let var = Variable "x"
  let eq1 = (var - 2) ** 2
  print eq1
  print $ take 10 $ solveIterate eq1 0.1

  let eq2 = var ** 2 - 2 * var - 8
  print eq2
  print (solveIterate eq2 0 !! 10, solveIterate eq2 2 !! 10)

  -- pi approximation 1
  let eq3 = sin var
  print $ take 10 $ solveIterate eq3 3

  -- pi approximation 2
  let eq4 = (exp (tan var) - 1) ** 3
  print eq4
  print $ take 10 $ solveIterate eq4 (-4)
