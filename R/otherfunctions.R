
### dmt function from mnormt package
dmt <- function (x, mean=rep(0,d), S, df = Inf, log = FALSE)
{
  if (df == Inf)  return(dmnorm(x, mean, S, log = log))
  d <- if(is.matrix(S)) ncol(S) else 1
  if (d==1) {
    y <- dt((x-mean)/sqrt(S), df=df, log=log)
    if(log) y <- (y - 0.5*logb(S)) else y <- y/sqrt(S)
    return(y)
  }
  x <- if (is.vector(x)) t(matrix(x)) else data.matrix(x)
  if (ncol(x) != d) stop("mismatch of dimensions of 'x' and 'varcov'")
  if (is.matrix(mean)) {if ((nrow(x) != nrow(mean)) || (ncol(mean) != d))
    stop("mismatch of dimensions of 'x' and 'mean'") }
  if(is.vector(mean)) mean <- outer(rep(1, nrow(x)), as.vector(matrix(mean,d)))
  X  <- t(x - mean)
  S.inv <- pd.solve(S, log.det=TRUE)
  Q <- colSums((S.inv %*% X) * X)
  logDet <- attr(S.inv, "log.det")
  logPDF <- (lgamma((df + d)/2) - 0.5 * (d * logb(pi * df) + logDet)
             - lgamma(df/2) - 0.5 * (df + d) * logb(1 + Q/df))
  if(log) logPDF else exp(logPDF)
}





