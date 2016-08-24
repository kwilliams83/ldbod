


#' @title Local Density-Based Outlier Detection using Subsampling with Approximate Nearest Neighbor Search
#' @description  This function computes local density-based outlier scores for input data.
#' @param X An n x p data matrix to compute outlier scores
#' @param k A vector of neighborhood sizes, k must be less than nsub
#' @param nsub Subsample size, nsub must be greater than k.  Usually nsub = 0.10*n or larger is recommended. Default is nsub = n
#' @param method Character vector specifying the local density-based method(s) to compute. User can specify more than
#' one method.  By default all methods are computed

#' @param ldf.param Vector of parameters for method LDF. h is the positive bandwidth parameter and c is a positive scaling constant.  Default values are h=1 and c=0.1
#' @param rkof.param Vector  parameters for method RKOF. C is the postive bandwidth paramter, alpha is a sensitiveity parameter in the interval [0,1],
#' and  sig2 is the variance parameter.  Default values are alpha=1, C=1, sig2=1
#' @param lpdf.param Vector of paramters for method LPDF.  cov.type is the covariance parameterization type,
#' which users can specifiy as either 'full' or 'diag'.  sigma2 is the positive regularization parameter, tmax is the maximum number of updates, and
#' v is the degrees of freedom for the multivariate t distribution.  Default values are cov.type = 'full',tmax=1, sigma2=1e-5, and v=1.
#' @param treetype Character vector specifiying tree method.  Either 'kd' or 'bd' tree may be specified.  Default is 'kd'.
#' Refer to documentation for RANN package.
#' @param eps Error bound.  Default is 0.0 which implies exact nearest neighgour search.  Refer to documentation for RANN package.
#' @param searchtype Character vector specifiying kNN search type. Default value is "standard". Refer to documentation for RANN package.
#' @param scale.data Logical value indicating to scale each feature of X using standard noramlization with mean 0 and standard deviation of 1
#' @details Computes the local density-based outlier scores for input data, X, referencing a random subsample of X. The subsampled
#' data set is constructed by drawning nsub samples from X without replacement.
#'
#' Four different methods can be implemented LOF, LDF, RKOF, and LPDF.  Each method specified returns densities and relative densities.
#' Methods LDF and RKOF uses guassian kernels, and method LDPF uses multivarite t distribution. Outlier scores returned are positive
#' except for lpde and lpdr which are log scaled densities (natural log). Score lpdr has shown to be highly sensitive to k.
#'
#' All kNN computations are carried out using the nn2() function from the RANN package. Multivariate t densities are
#' computed using the dmt() function from the mnormt package.  Refer to specific packages for more details.  Note: all
#' neighborhoods are strickly of size k; therefore, the algorithms for LOP, LDF, and RKOF are not exact implementations, but
#' algorithms are similiar for most situation and are equivalent when distance to k-th nearest neighbor is unique.  If there are many
#' many duplicate data points, then implementation of algorithms could lead to dramatically different (positive or negative) results than those that allow
#' neighborhood sizes larger than k, especially if k is relatively small.  Removing duplicates is recommended before computing
#' outlier scores unless there is good reason to keep them.
#'
#' The algorithm can be used to compute an ensemble of unsupervised outlier scores by using multiple k values
#' and iterating over multiple subsamples.
#'
#'
#' @return
#' A list of length 9 with the elements:
#'
#' lrd --An n x length(k) matrix where each column vector represents the local reachabiility denity (LRD) outlier scores for each specifed k value.  Smaller values indicate a point in more outlying.
#'
#' lof   --An n x length(k) matrix where each column vector represents the local outlier factor (LOF) outlier scores for each specifed k value. Larger values indicate a point in more outlying.
#'
#' lde   --An n x length(k) matrix where each column vector represents the local density estimate (LDE) outlier scores for each specifed k value. Smaller values indicate a point in more outlying.
#'
#' ldf   --An n x length(k) matrix where each column vector represents the local density factor (LDF) outlier scores for each specifed k value. Larger values indicate a point in more outlying.
#'
#' kde   --An n x length(k) matrix where each column vector represents the kernel density estimate (KDE) outlier scores for each specifed k value. Smaller values indicate a point in more outlying.
#'
#' rkof   --An n x length(k) matrix where each column vector represents the robust kernel density factor (RKOF) outlier scores for each specifed k value. Larger values indicate a point in more outlying.
#'
#' lpde   --An n x length(k) matrix where each column vector represents the local parametric density estimate (LPDE) outlier scores for each specifed k value on log scale. Smaller values indicate a point in more outlying.
#'
#' lpdf  --An n x length(k) matrix where each column vector represents the local parametric density factor (LPDF) outlier scores for each specifed k value. Smaller values indicate a point in more outlying.
#'
#' lpdr   --An n x length(k) matrix where each column vector represents the local parametric density ratio (LPDR) outlier scores for each specifed k value. Smaller values indicate a point in more outlying.
#' LPDR is typically used to detect groups of outliers.
#'
#' If a method is not specified then returns NULL
#'
#' @references M. M. Breunig, H-P. Kriegel, R.T. Ng, and J. Sander (2000). LOF: Identifying density-based local outliers.  In Proc. of ACM
#' International Conference on Knowledge Discovery and Data Mining, 93-104.
#'
#' L. J. Latecki, A. Lazarevic, and D. Pokrajac (2007). Outlier Detection with kernel density funcions.  In Proc. of Machine Learning and Data
#' Mining in Pattern Recognition, 61-75
#'
#' J. Gao, W. Hu, Z. Zhang, X. Zhang, and O. Wu (2011). RKOF: Robust kernel-based local outlier detection. In Proc. of Advances in Knowledge Discovery and
#' Data Mining, 270-283.
#'
#' K. Williams (2016).  Local parametric density-based outlier deteciton and ensemble learning with application to malware detection. PhD dissertation,
#' The University of Texas at San Antonio. Unpublished manuscript
#' @examples
#' # 500 x 2 data matrix
#' X <- matrix(rnorm(1000),500,2)
#'
#' # five outliers
#' outliers <- matrix(c(rnorm(2,20),rnorm(2,-12),rnorm(2,-8),rnorm(2,-5),rnorm(2,9)),5,2)
#'  X <- rbind(X,outliers)
#'
#'# compute outlier scores without subsampling for all methods
#' scores <- ldbod(X, k=50)
#'
#' head(scores$lrd); head(scores$rkof)
#'
#' # plot data and highlight top 5 outliers retured by lof
#' plot(X)
#' points(X[order(scores$lof,decreasing=TRUE)[1:5],],col=2)
#'
#' # plot data and highlight top 5 outliers retured by outlier score lpde
#' plot(X)
#' points(X[order(scores$lpde,decreasing=FALSE)[1:5],],col=2)
#'
#'
#'# compute outlier scores for k= 10,20 with 10% subsampling for methods 'lof' and 'lpdf'
#' scores <- ldbod(X, k = c(10,20), nsub = 0.10*nrow(X), method = c('lof','lpdf'))
#' scores <- ldbod(X, k = c(10,20),nsub = 0.10*nrow(X), method = c('lof','lpdf'))

#'
#' # plot data and highlight top 5 outliers retuned by lof for k=20
#' plot(X)
#' points(X[order(scores$lof[,2],decreasing=TRUE)[1:5],],col=2)

#'
#'
#' @export
#'
ldbod <- function(X, k = c(10,20), nsub = nrow(X), method = c('lof','ldf','rkof','lpdf'),
                  ldf.param = c(h = 1, c = 0.1),
                  rkof.param = c(alpha = 1, C = 1, sig2 = 1),
                  lpdf.param = c(cov.type = 'full',sigma2 = 1e-5, tmax=1, v=1),
                  treetype='kd',searchtype='standard',eps=0.0,
                  scale.data=TRUE)
{

  if(is.null(k)) stop('k is missing')

  if(!is.numeric(k)) stop('k is not numeric')


  # coerce X to class matrix
  X <- as.matrix(X)

  if(!is.numeric(X)) stop('X contains non-numeric data type')

  if(!is.matrix(X)) stop('X must be of class matrix')

  k <- as.integer(k)

  nsub <- as.integer(nsub)

  # check max k less than than nsub-1
  kmax <-  max(k)
  len.k <- length(k)
  if(kmax > nsub - 1 ){ stop('k is greater than nsub') }

  if(is.null(nsub)) (nsub <- nrow(X))

  nsub  <- as.integer(nsub)
  kmax  <-  max(k)
  len.k <- length(k)


  if(kmax > nsub - 1 )
    stop('k is greater than nsub')

  if(min(k) < 2 )
    stop('k must be greater than 1')





  # number of rows of X
  n <- nrow(X)
  # number of columns of X
  p <- ncol(X)

  # check that nsub <= n
  if(nsub>n){ nsub=n }
  if(nsub<10){ nsub=10 }

  # subsample ids without replacement
  sub_sample_ids = sample(1:n,nsub)


  # scale X
  if(scale.data){  X <- scale(X) }

  # subsample X, Y will subsample of X
  Y <-  as.matrix(X[sub_sample_ids,])



  # compute distance (euclidean) matrix between X and Y and returns kNN ids and kNN distances #
  knn <- nn2(data = Y,query = X,k = kmax+1,treetype=treetype, searchtype=searchtype, eps=eps)

  # kNN id matrix
  knn_ids <- knn$nn.idx[,-1]

  # kNN distance matrix
  knn_dist_matrix <- knn$nn.dists[,-1]



  # define storeage matrix for each outlier score ................
  if('lof'%in%method){
    store_lrd <- matrix(NA,n,len.k)
    store_lof <- matrix(NA,n,len.k)
  }else{
    store_lrd <- NULL
    store_lof <- NULL
  }

  if('ldf'%in%method){
    store_lde <- matrix(NA,n,len.k)
    store_ldf <- matrix(NA,n,len.k)
  }else{
    store_lde <- NULL
    store_ldf <- NULL
  }

  if('rkof'%in%method){
    store_kde <- matrix(NA,n,len.k)
    store_rkof <- matrix(NA,n,len.k)
  }else{
    store_kde <- NULL
    store_rkof <- NULL
  }

  if('lpdf'%in%method){
    store_lpde <- matrix(NA,n,len.k)
    store_lpdf <- matrix(NA,n,len.k)
    store_lpdr <- matrix(NA,n,len.k)
  }else{
    store_lpde <- NULL
    store_lpdf <- NULL
    store_lpdr <- NULL
  }

  # initialize
  ii <- 0

  ### compute lof and lrd for each k value specified by user
  for(kk in k){

    ii <- ii+1

    # distance to kth nearest neighbor for X to Y
    dist_k <- knn_dist_matrix[,kk]


    ## Reachability distance matrix if outlier score lrd, lof, lde, or ldf has been specified
    if(sum(method%in%c('lof','ldf'))>0){

      # arrange dist_k relative to subsample data set
      dist_k_rel_to_test = t(apply(knn_ids[,1:kk],1,function(x)dist_k[sub_sample_ids[x]]))

      # compute reachability distances between X and kNN of X relative to subsample data set
      # returns reachability distance matrix of size n x kk
      reach_dist_matrix_test <- t(apply( cbind(knn_dist_matrix[,1:kk] , dist_k_rel_to_test),1,function(x){

        x1 = x[1:kk]
        x2 = x[(kk+1):(2*kk)]
        reach_dist = apply(rbind(x1,x2),2,max)
        return(reach_dist)

      }))


      ############  compute outlier scores lrd and lof  ############
      ############  compute outlier scores lrd and lof  ############
      ############  compute outlier scores lrd and lof  ############
      if('lof'%in%method){

        #lof.scores <- lof.fun(kk,n,p, sub_sample_ids, knn_ids, reach_dist_matrix_test)


        # compute local reachability density for each point in X
        lrd <- 1/(apply(reach_dist_matrix_test,1,mean)+1e-200)

        # compute local outlier factor for each point in X
        lof <- apply(knn_ids[,1:kk],1,function(x)mean(lrd[sub_sample_ids[x]]))/lrd

        # store lof and lrd for each k
        store_lrd[,ii] <- lrd#lof.scores$lrd
        store_lof[,ii] <- lof#lof.scores$lof

      }# end if statement for lof

      ############  compute outlier scores lde and ldf  ############
      ############  compute outlier scores lde and ldf  ############
      ############  compute outlier scores lde and ldf  ############
      if('ldf'%in%method){

        # paramters
        h <- ldf.param[names(ldf.param)=='h']
        c <- ldf.param[names(ldf.param)=='c']

        #ldf.scores <- ldf.fun(kk,n,p, knn_ids, sub_sample_ids, dist_k, reach_dist_matrix_test, h,c)

        lde <- sapply(1:n,function(id){
          mean(1/((2*pi)^(p/2))*1/(h*dist_k[sub_sample_ids[knn_ids[id,1:kk]]])^p*
                 exp(-(.5*reach_dist_matrix_test[id,]^2)/(h*dist_k[sub_sample_ids[knn_ids[id,1:kk]]])^2))+1e-200
        })

        ldf <- sapply(1:n,function(id){
          mean.lde = mean(lde[sub_sample_ids[knn_ids[id,1:kk]]])
          ldf = mean.lde/(lde[id]+c*mean.lde)
          return(ldf)
        })

        # store lof and lrd for each k
        store_lde[,ii] <- lde#ldf.scores$lde
        store_ldf[,ii] <- ldf#ldf.scores$ldf

      }# end if statement for ldf


    }# end if statement for reachability distance calculations




    ############  compute outlier scores kde and rkof  ############
    ############  compute outlier scores kde and rkof  ############
    ############  compute outlier scores kde and rkof  ############
    if('rkof'%in%method){

      # parameters
      alpha <- rkof.param[names(rkof.param)=='alpha']
      C     <- rkof.param[names(rkof.param)=='C']
      sig2  <- rkof.param[names(rkof.param)=='sig2']

      # compute kde, wde, and rkof
      #rkof.scores <- rkof.fun(kk,n,p,knn_ids, sub_sample_ids, dist_k, knn_dist_matrix,alpha, C, sig2)

      ## compute kde for each point in X
      kde <-  sapply(1:n, function(id){

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

      # store kde and rkof for each k
      store_kde[,ii] <- kde#rkof.scores$kde
      store_rkof[,ii] <- rkof#rkof.scores$rkof

    }# end if statement for rkof




    ############  compute outlier scores lpde, lpdf, and lpdr  ############
    ###########  compute outlier scores lpde, lpdf, and lpdr  ############
    ###########  compute outlier scores lpde, lpdf, and lpdr  ############
    if('lpdf'%in%method){

      # parameters
      cov.type<- lpdf.param[names(lpdf.param)=='cov.type']
      tmax   <- as.numeric(lpdf.param[names(lpdf.param)=='tmax'])
      sigma2 <- as.numeric(lpdf.param[names(lpdf.param)=='sigma2'])
      v      <- as.numeric(lpdf.param[names(lpdf.param)=='v'])

      # sub function for computing lpde, lpdf, lpdr
      #lpdf.scores <- lpdf.fun(X,Y,kk,n,p,cov.type, sigma2, tmax, v, knn_ids, sub_sample_ids, dist_k, reach_dist_matrix_test)



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


      store_lpde[,ii] <- lpde#lpdf.scores$lpde
      store_lpdf[,ii] <- lpdf#lpdf.scores$lpdf
      store_lpdr[,ii] <- lpdr#lpdf.scores$lpdr


    }# end if statement for lpdf



  }# k loop

  # return all outlier scores
  return(list(lrd = store_lrd,
              lof = store_lof,
              lde = store_lde,
              ldf = store_ldf,
              kde = store_kde,
              rkof = store_rkof,
              lpde = store_lpde,
              lpdf = store_lpdf,
              lpdr = store_lpdr
  )
  )



} # end ldbod function










