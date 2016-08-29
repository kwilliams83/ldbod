


#' @title Local Density-Based Outlier Detection using Reference Data with Approximate Nearest Neighbor Search
#' @description  This function computes local density-based outlier scores for input data and user specified reference set.
#' @param X An n x p data matrix to compute outlier scores
#' @param Y An m x p reference data matrix.
#' @param k A vector of neighborhood sizes, k must be less than m.
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
#' @param scale.data Logical value indicating to scale each feature of X using standard noramlization based on mean and standard deviation for features of Y.
#'
#' @details Computes local density-based outlier scores for input data, X, referencing data Y.  For semi-supervised outlier detection Y would be a set of "normal"
#' reference points; otherwise, Y can be any other set of reference points of interest. This allows users the flexibility to reference other data sets besides X or
#' a subset of X.
#' Four different methods can be implemented LOF, LDF, RKOF, and LPDF.  Each method specified returns densities and relative densities.
#' Methods LDF and RKOF uses guassian kernels, and method LDPF uses multivarite t distribution.
#' Outlier scores returned are non-negative except for lpde adn lpdr which are log scaled densities (natural log). Note: Outlier score
#' lpdr is strictly designed for unsupervised outlier detection and should not be used in the semi-supervised setting.
#' Refer to references for
#' more details about each method.
#'
#' All kNN computations are carried out using the nn2() function from the RANN package. Multivariate t densities are
#' computed using the dmt() function from the mnormt package.  Refer to specific packages for more details.  Note: all
#' neighborhoods are strickly of size k; therefore, the algorithms for LOP, LDF, and RKOF are not exact implementations, but
#' algorithms are similiar for most situation and are equivalent when distance to k-th nearest neighbor is unique.  If there are many
#' many duplicate data points in Y, then implementation of algorithms could lead to dramatically different (positive or negative) results than those that allow
#' neighborhood sizes larger than k, especially if k is relatively small.  Removing duplicates is recommended before computing
#' outlier scores unless there is good reason to keep them.
#'
#' #' The algorithm can be used to compute an ensemble of unsupervised outlier scores by using multiple k values
#' and multiple iterations of reference data.
#'
#' @return
#' A list of length 9 with the elements:
#'
#' lrd --An n x length(k) matrix where each column vector represents outlier scores for each specifed k value.  Smaller values indicate a point in more outlying.
#'
#' lof   --An n x length(k) matrix where each column vector represents outlier scores for each specifed k value. Larger values indicate a point in more outlying.
#'
#' lde   --An n x length(k) matrix where each column vector represents outlier scores for each specifed k value. Smaller values indicate a point in more outlying.
#'
#' ldf   --An n x length(k) matrix where each column vector represents outlier scores for each specifed k value. Larger values indicate a point in more outlying.
#'
#' kde   --An n x length(k) matrix where each column vector represents outlier scores for each specifed k value. Smaller values indicate a point in more outlying.
#'
#' rkof   --An n x length(k) matrix where each column vector represents outlier scores for each specifed k value. Larger values indicate a point in more outlying.
#'
#' lpde   --An n x length(k) matrix where each column vector represents outlier scores for each specifed k value. Smaller values indicate a point in more outlying.
#'
#' lpdf  --An n x length(k) matrix where each column vector represents outlier scores for each specifed k value. Smaller values indicate a point in more outlying.
#'
#' lpdr   --An n x length(k) matrix where each column vector represents outlier scores for each specifed k value. Smaller values indicate a point in more outlying.
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
#' K. Williams (2016).  Local parametric density-based outlier deteciton and ensemble learning with applications to malware detection. PhD dissertation,
#' University of Texas at San Antonio.
#' @examples
#' # 500 x 2 data matrix
#' X <- matrix(rnorm(1000),500,2)
#' Y <- X
#' # five outliers
#' outliers <- matrix(c(rnorm(2,20),rnorm(2,-12),rnorm(2,-8),rnorm(2,-5),rnorm(2,9)),5,2)
#'  X <- rbind(X,outliers)
#'
#'# compute outlier scores referencing Y for all methods
#' scores <- ldbod.ref(X,Y, k=50)
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
#'  # compute outlier scores for k= 10,20 referencing Y for methods 'lof' and 'lpdf'
#' scores <- ldbod.ref(X,Y, k = c(10,20), method = c('lof','lpdf'))
#'
#' # plot data and highlight top 5 outliers retuned by lof for k=20
#' plot(X)
#' points(X[order(scores$lof[,2],decreasing=TRUE)[1:5],],col=2)
#'
#'
#' @importFrom stats sd cov.wt sd dt
#' @importFrom RANN nn2
#' @importFrom mnormt dmt pd.solve dmnorm

#' @export
ldbod.ref <- function(X , Y , k = c(10,20), method = c('lof','ldf','rkof','lpdf'),
                      ldf.param = c(h = 1, c = 0.1),
                      rkof.param = c(alpha = 1, C = 1, sig2 = 1),
                      lpdf.param = c(cov.type = 'full', sigma2 = 1e-5, tmax=1, v=1),
                      treetype='kd', searchtype='standard',eps=0.0,
                      scale.data=TRUE)
{

  if(is.null(k)) stop('k is missing')

  if(!is.numeric(k)) stop('k is not numeric')

  # coerce X and Y to class matrix
  X <- as.matrix(X)
  Y <- as.matrix(Y)

  if(!is.numeric(X)) stop('the data matrix X contains non-numeric data type')

  if(!is.numeric(Y)) stop('the data matrix Y contains non-numeric data type')




  if(!is.matrix(X)) stop('X must be of class matrix')


  if(!is.matrix(Y)) stop('Y must be of class matrix')







  # number of rows of X
  n <- nrow(X)
  # number of rows of Y
  m <- nrow(Y)
  # number of columns of X
  p <- ncol(X)


  # scale X by mean and sd of features of Y
  if(scale.data){

    scale.y <- apply(Y,2,function(x)c(mean(x),sd(x)))
    Y       <- scale(Y)
    X       <- scale(X,center=scale.y[1,],scale=scale.y[2,])

  }

  # check max k less than m
  kmax  <-  max(k)
  len.k <- length(k)
  if(kmax > m-1 ) stop('k is greater than size of reference set Y')
  if(min(k) < 2 ) stop('k must be greater than 1')

  # compute distance (euclidean) matrix between X and Y and returns kNNs ids and kNNs distances
  knn <- nn2(data=Y,query=X,k = kmax+1,treetype=treetype,searchtype=searchtype)
  knn_ids <- knn$nn.idx[,-1]
  knn_dist_matrix <- knn$nn.dists[,-1]

  # compute distance (euclidean) matrix between Y and Y and returns kNNs ids and kNNs distances
  knn_train <- nn2(data=Y,query=Y,k = kmax+1,treetype=treetype,searchtype=searchtype)
  knn_ids_train <- knn_train$nn.idx[,-1]
  knn_dist_matrix_train <- knn_train$nn.dists[,-1]

  # define storeage matrix for each outlier score
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
    # distance to kth nearest neighbor for Y to Y
    dist_k_train <- knn_dist_matrix_train[,kk]


    ## Reachability distance matrix if outlier score lrd, lof, lde, or ldf has been specified
    if(sum(method%in%c('lof','ldf'))>0){

      # arrange k_dist_train relative to X
      dist_k_rel_to_test <-  t(apply(knn_ids[,1:kk],1,function(x)dist_k_train[x]))

      # arrange k_dist_train relative to Y
      dist_k_rel_to_train <-  t(apply(knn_ids_train[,1:kk],1,function(x)dist_k_train[x]))

      # compute reachability distances between X and kNN of X relative to Y
      reach_dist_matrix_test <- t(apply( cbind(knn_dist_matrix[,1:kk] , dist_k_rel_to_test),1,function(x){

        x1 = x[1:kk]
        x2 = x[(kk+1):(2*kk)]
        ## reach_dist(x,y) = max( dist(x,y), k_dist(y))
        reach_dist = apply(rbind(x1,x2),2,max)
        return(reach_dist)

      }))


      # compute reachability distances between Y and kNN of Y relative to Y
      reach_dist_matrix_train <- t(apply( cbind(knn_dist_matrix_train[,1:kk] , dist_k_rel_to_train),1,function(x){

        x1 = x[1:kk]
        x2 = x[(kk+1):(2*kk)]
        ## reach_dist(x,y) = max( dist(x,y), k_dist(y))
        reach_dist = apply(rbind(x1,x2),2,max)
        return(reach_dist)

      }))

    }# end if statement for reachability distance calculations


    ############  compute outlier scores lrd and lof  ############
    ############  compute outlier scores lrd and lof  ############
    ############  compute outlier scores lrd and lof  ############
    if('lof'%in%method){

      # compute local reachability density for test points
      #lof.scores <- lof.ref.fun(kk, knn_ids,reach_dist_matrix_test,reach_dist_matrix_train)

      # compute local reachability density for test points
      lrd <- 1/(apply(reach_dist_matrix_test,1,mean)+1e-200)
      # compute local reachability density for train points
      lrd_train <- 1/(apply(reach_dist_matrix_train,1,mean)+1e-200)

      # compute local outlier factor for test points
      lof <- apply(knn_ids[,1:kk],1,function(x)mean(lrd_train[x]))/lrd

      # store lof and lrd for each k
      store_lrd[,ii] <- lrd#lof.scores$lrd
      store_lof[,ii] <- lof#lof.scores$lof

    }# end if statement for lof

    ############  compute outlier scores lde and ldf  ############
    ############  compute outlier scores lde and ldf  ############
    ############  compute outlier scores lde and ldf  ############
    if('ldf'%in%method){

      # parameters
      h <- ldf.param[names(ldf.param)=='h']
      c <- ldf.param[names(ldf.param)=='c']

      ## returns lde and ldf
      #ldf.scores <- ldf.ref.fun(n, m, p, kk, knn_ids, knn_ids_train, dist_k_train, reach_dist_matrix_test, reach_dist_matrix_train, h, c)

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


      # store lof and lrd for each k
      store_lde[,ii] <- lde#ldf.scores$lde
      store_ldf[,ii] <- ldf#ldf.scores$ldf

    }# end if statement for ldf


    ############  compute outlier scores kde and rkof  ############
    ############  compute outlier scores kde and rkof  ############
    ############  compute outlier scores kde and rkof  ############
    if('rkof'%in%method){

      alpha <- rkof.param[names(rkof.param)=='alpha']
      C     <- rkof.param[names(rkof.param)=='C']
      sig2  <- rkof.param[names(rkof.param)=='sig2']

      #rkof.scores <- rkof.ref.fun(n, m, p, kk, knn_ids, knn_ids_train, dist_k_train,
      #                            knn_dist_matrix, knn_dist_matrix_train, alpha, C, sig2)

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


      # store lof and lrd for each k
      store_kde[,ii] <- kde#rkof.scores$kde
      store_rkof[,ii] <- rkof#rkof.scores$rkof

    }# end if statement for rkof




    ############  compute outlier scores lpde, lpdf, and lpdr  ############
    ###########  compute outlier scores lpde, lpdf, and lpdr  ############
    ###########  compute outlier scores lpde, lpdf, and lpdr  ############
    if('lpdf'%in%method){

      cov.type <- lpdf.param[names(lpdf.param)=='cov.type']
      tmax   <- as.numeric(lpdf.param[names(lpdf.param)=='tmax'])
      sigma2 <- as.numeric(lpdf.param[names(lpdf.param)=='sigma2'])
      v      <- as.numeric(lpdf.param[names(lpdf.param)=='v'])


      #lpdf.scores <- lpdf.ref.fun(X, Y, n, m, p, kk, knn_ids, knn_ids_train, dist_k_train,
       #                            knn_dist_matrix, knn_dist_matrix_train, cov.type, tmax, sigma2, v)

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
          log1p(dens[id])/sum(weights*log1p(dens_train[knn_ids[id,1:kk]]))

        })




        ### compute updated weight function for training data, Y
        R_train <- sapply(1:m,function(id){

          # compute weights
          weights = 1/(dist_k_train[knn_ids_train[id,1:kk]]^2+1e-20)
          weights = weights/sum(weights)

          # comptue ratio of density to weighted neighborhood density
          log1p(dens_train[id])/sum(weights*log1p(dens_train[knn_ids_train[id,1:kk]]))

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


      store_lpde[,ii] <- lpde#lpdf.scores$lpde
      store_lpdf[,ii] <- lpdf#lpdf.scores$lpdf
      store_lpdr[,ii] <- lpdr#lpdf.scores$lpdr

    }# end if statement for lpdf



  }### k loop

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






