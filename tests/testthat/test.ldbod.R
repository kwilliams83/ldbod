


### test that ldbod is working properly

## one dimensional matrix
X <-  rnorm(500)
## compute ldbod scores
s1 <- ldbod(X, k=10, nsub = 250)


## two dimensional matrix
X <- matrix(rnorm(1000),500,2)
## compute ldbod scores
s1 <- ldbod(X, k=10, nsub = 250)



## three dimensional matrix
X <- matrix(rnorm(1500),500,3)
## compute ldbod scores
s1 <- ldbod(X, k=10, nsub = 250)

