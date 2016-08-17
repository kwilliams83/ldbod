


#### sub-routine functions #####




# sub function for computing lrd and lof
lof.fun <- function(kk, n,p,sub_sample_ids, knn_ids, reach_dist_matrix_test)
{

  # compute local reachability density for each point in X
  lrd <- 1/(apply(reach_dist_matrix_test,1,mean)+1e-200)

  # compute local outlier factor for each point in X
  lof <- apply(knn_ids[,1:kk],1,function(x)mean(lrd[sub_sample_ids[x]]))/lrd

  return(list(lrd=lrd, lof=lof))

}





# sub function for computing lde and ldf
ldf.fun <- function(kk,n,p, knn_ids, sub_sample_ids, dist_k, reach_dist_matrix_test, h,c)
{


  lde <- sapply(1:n,function(id){
    mean(1/((2*pi)^(p/2))*1/(h*dist_k[sub_sample_ids[knn_ids[id,1:kk]]])^p*
           exp(-(.5*reach_dist_matrix_test[id,]^2)/(h*dist_k[sub_sample_ids[knn_ids[id,1:kk]]])^2))+1e-200
  })

  ldf <- sapply(1:n,function(id){
    mean.lde = mean(lde[sub_sample_ids[knn_ids[id,1:kk]]])
    ldf = mean.lde/(lde[id]+c*mean.lde)
    return(ldf)
  })

  return(list(lde=lde,ldf=ldf))
}




# sub function for computing kde and rkof
rkof.fun <- function(kk,n,p,knn_ids, sub_sample_ids, dist_k, knn_dist_matrix,alpha, C, sig2)
{

  n <- nrow(X)
  p <- ncol(X)
  ## compute kde for each point in X
  kde <-  sapply(1:n,function(id){

    mean(1/(2*pi)^(p/2)*1/(C*dist_k[sub_sample_ids[knn_ids[id,1:kk]]]^alpha)^2*
           exp(-.5*knn_dist_matrix[id,1:kk]^2/(C*dist_k[sub_sample_ids[knn_ids[id,1:kk]]]^alpha)))+1e-200

  })

  ## compute wde for each point in X
  wde <-  sapply(1:n,function(id){

    weights = exp(-(dist_k[sub_sample_ids[knn_ids[id,1:kk]]]/min(dist_k[sub_sample_ids[knn_ids[id,1:kk]]])-1)^2/(2*sig2))

    weights = weights/sum(weights)
    wde = sum(weights*kde[sub_sample_ids[knn_ids[id,1:kk]]])

  })


  ## compute rkof
  rkof <- wde/kde

  return(list(kde=kde, rkof=rkof))

}










# sub function for computing lpde, lpdf, lpdr
lpdf.fun <- function(X,Y,n,p,kk, cov.type, sigma2, tmax, v, knn_ids, sub_sample_ids, dist_k, reach_dist_matrix_test)
{


  # max iterations
  tmax <- tmax+1
  # identity matrix
  II <- diag(1,p,p)
  # regularization matrix
  reg <- sigma2*II

  # matrices to store densities and weights
  store_dens <- matrix(NA,n,tmax)
  store_R <- matrix(NA,n,tmax)

  # compute density and weight function, R, for each iteration in 1:tmax
  for(t in 1:tmax){

    #compute multivarite t density for each point in X referencing subsample data set Y
    dens <-  sapply(1:n,function(id){

      # test point
      x = X[id,]

      # k nearest neighborhood to compute weighted location and scatter
      hood = as.matrix(Y[knn_ids[id,1:kk],])

      # define weights
      if(t==1){weights = rep(1/kk,kk)}
      if(t>1){weights  = R[sub_sample_ids[knn_ids[id,1:kk]]] }

      # normalize weights to sum to 1
      weights <-  weights/sum(weights)

      # comptue weighted sample mean and sample covariance matrix  if cov.type='full'
      if(cov.type=='full'){
        covwt   = cov.wt(hood,wt=weights,method='ML')
        center  = covwt$center
        scatter = covwt$cov+reg

      }

      # comptue weighted sample mean and sample covariance matrix if cov.type='diag'
      if(cov.type=='diag'){
        center     = colSums(weights * hood)
        center.mat = matrix(center,kk,p,byrow=T)
        vars       = colSums(weights*(hood-center.mat)^2)
        scatter    = diag(vars,p,p)+reg
      }


      # compute multivaritae t density with degrees of freedom v
      density =  dmt(x,mean=center,S=scatter,df=v)+1e-200

    })


    # compute updated weight function, R, for each point in X
    R <- sapply(1:n,function(id){

      # compute weights
      weights = 1/(dist_k[sub_sample_ids[knn_ids[id,1:kk]]]^2+1e-20)
      weights = weights/sum(weights)

      # comptue ratio of density to weighted neighborhood density
      log1p(dens[id])/sum(log1p(dens[sub_sample_ids[knn_ids[id,1:kk]]]))

    })


    # store all densities and R
    store_dens[,t] <- dens
    store_R[,t] <- R

    ## update weight rule
    if(t>2){

      R <- apply(store_R[,2:t],1,max)
    }


  }### end t loop

  # compute outlier scores lpde, lpdf, lpdr
  lpde <- apply(store_dens,1,function(x)log(mean(x[-1])))
  lpdf <- apply(store_R,1,function(x)mean(x[-1]))
  lpdr <- lpde - log(store_dens[,1])

  return(list(lpde=lpde,lpdf=lpdf,lpdr=lpdr))



}


















