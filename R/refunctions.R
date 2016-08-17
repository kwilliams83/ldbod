
## sub-routine funcitons for ldbod.ref



##  returns lrd and lof
lof.ref.fun <- function(kk, knn_ids,reach_dist_matrix_test,reach_dist_matrix_train){

  # compute local reachability density for test points
  lrd <- 1/(apply(reach_dist_matrix_test,1,mean)+1e-200)
  # compute local reachability density for train points
  lrd_train <- 1/(apply(reach_dist_matrix_train,1,mean)+1e-200)

  # compute local outlier factor for test points
  lof <- apply(knn_ids[,1:kk],1,function(x)mean(lrd_train[x]))/lrd

  return(list(lrd=lrd,lof=lof))


}







## returns lde and ldf
ldf.ref.fun <- function(n,m,p,kk,knn_ids,knn_ids_train,dist_k_train,reach_dist_matrix_test,reach_dist_matrix_train,h,c){



  ## compute local density estimate for test and train data sets
  lde <-sapply(1:n,function(id)mean(1/((2*pi)^(p/2))*1/(h*dist_k_train[knn_ids[id,1:kk]])^p*
                                exp(-(.5*reach_dist_matrix_test[id,]^2)/(h*dist_k_train[knn_ids[id,1:kk]])^2))+1e-200)

  lde_train <- sapply(1:m,function(id)mean(1/((2*pi)^(p/2))*1/(h*dist_k_train[knn_ids_train[id,1:kk]])^p*
                                       exp(-(.5*reach_dist_matrix_train[id,]^2)/(h*dist_k_train[knn_ids_train[id,1:kk]])^2))+1e-200)

  ## compute local density factor for test
  ldf <- sapply(1:n,function(id){
    mean.lde = mean(lde_train[knn_ids[id,1:kk]])
    ldf = mean.lde/(lde[id]+c*mean.lde)
    return(ldf)
  })

  return(list(lde=lde,ldf=ldf))


}




## returns kde and rkof
rkof.ref.fun <- function(n,m,p,kk,knn_ids, knn_ids_train, dist_k_train, knn_dist_matrix,knn_dist_matrix_train,alpha,C,sig2)

  {



  ## compute kde for test set
  kde <-  sapply(1:n,function(id){

    mean(1/(2*pi)^(p/2)*1/(C*dist_k_train[knn_ids[id,1:kk]]^alpha)^2*
           exp(-.5*knn_dist_matrix[id,1:kk]^2/(C*dist_k_train[knn_ids[id,1:kk]]^alpha)))+1e-200

  })

  ## compute kde for train set
  kde_train <-  sapply(1:m,function(id){

    mean(1/(2*pi)^(p/2)*1/(C*dist_k_train[knn_ids_train[id,1:kk]]^alpha)^2*
           exp(-.5*knn_dist_matrix_train[id,1:kk]^2/(C*dist_k_train[knn_ids_train[id,1:kk]]^alpha)))+1e-200

  })

  ## compute wde for test set
  wde <-  sapply(1:n,function(id){

    weights = exp(-(dist_k_train[knn_ids[id,1:kk]]/min(dist_k_train[knn_ids[id,1:kk]])-1)^2/(2*sig2))

    weights = weights/sum(weights)

    wde = sum(weights*kde_train[knn_ids[id,1:kk]])
  })

  ## compute rkof for test set
  rkof <- wde/kde

  return(list(kde=kde,rkof=rkof))



}




lpdf.ref.fun <- function(X, Y, n, m, p, kk, knn_ids, knn_ids_train, dist_k_train,
                         knn_dist_matrix, knn_dist_matrix_train, cov.type, tmax, sigma2, v)
{



  tmax <- tmax+1
  II <- diag(1,p,p)
  reg <- sigma2*II

  # compute density and weight function, R, for each iteration in 1:tmax
  store_dens <- matrix(NA,n,tmax)
  store_R <- matrix(NA,n,tmax)
  store_R_train <- matrix(NA,m,tmax)

  for(t in 1:tmax){

    ## compute multivarite t density for test data relative to training data
    dens <-  sapply(1:n,function(id){

      # test point
      x = X[id,]
      # neighborhood to compute weighted location and scatter
      hood = as.matrix(Y[knn_ids[id,1:kk],])

      # define weights
      if(t==1){weights = rep(1/kk,kk)}
      if(t>1){weights = R_train[knn_ids[id,1:kk]]}

      # normalize weights to sum to 1
      weights = weights/sum(weights)

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
      density = dmt(x,mean=center,S=scatter,df=v)+1e-200

    })


    ## compute multivarite t density for train data Y
    dens_train <-  sapply(1:m,function(id){

      # test point
      y = Y[id,]
      # neighborhood to compute weighted location and scatter
      hood = as.matrix(Y[knn_ids_train[id,1:kk],])

      # define weights
      if(t==1){weights = rep(1/kk,kk)}
      if(t>1){weights=R_train[knn_ids_train[id,1:kk]]}

      # normalize weights to sum to 1
      weights = weights/sum(weights)

      # compute location and covariance matrix if cov.type='full'
      if(cov.type=='full'){
        covwt   = cov.wt(hood,wt=weights,method='ML')
        center  = covwt$center
        scatter = covwt$cov+reg

      }

      # compute location and covariance matrix if cov.type='diag'
      if(cov.type=='diag'){
        center     = colSums(weights * hood)
        center.mat = matrix(center,kk,p,byrow=T)
        vars       = colSums(weights*(hood-center.mat)^2)
        scatter    = diag(vars,p,p)+reg
      }


      # compute multivaritae t density with degrees of freedom v
      density = dmt(y,mean=center,S=scatter,df=v)+1e-200

    })



    ### compute updated weight function
    R <- sapply(1:n,function(id){

      # compute weights
      weights = 1/(dist_k_train[knn_ids[id,1:kk]]^2+1e-20)
      weights = weights/sum(weights)
      # comptue ratio of density to weighted neighborhood density
      log1p(dens[id])/sum(log1p(dens_train[knn_ids[id,1:kk]]))

    })

    ### compute updated weight function for training data, Y
    R_train <- sapply(1:m,function(id){

      # compute weights
      weights = 1/(dist_k_train[knn_ids_train[id,1:kk]]^2+1e-20)
      weights = weights/sum(weights)
      # comptue ratio of density to weighted neighborhood density
      log1p(dens_train[id])/sum(log1p(dens_train[knn_ids_train[id,1:kk]]))

    })

    # store all densities and R
    store_dens[,t] <- dens
    store_R[,t] <- R
    store_R_train[,t] <- R_train

    ## update weight rule
    if(t>2){

      R <- apply(store_R[,2:t],1,max)
      R_train <- apply(store_R_train[,2:t],1,max)
    }


  }# end t loop

  ### compute outlier scores lpde, lpdf, lpdr
  lpde <- apply(store_dens,1,function(x)log(mean(x[-1])))
  lpdf <- apply(store_R,1,function(x)mean(x[-1]))
  lpdr <- lpde-log(store_dens[,1])

  return(list(lpde=lpde,lpdf=lpdf,lpdr=lpdr))





}































