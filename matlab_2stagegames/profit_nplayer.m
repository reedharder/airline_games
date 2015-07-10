function g=profit_nplayer(p,f,M,S,C,alpha,beta,N,i,p_i)
    p(i)=p_i;
    num=exp(alpha*log(f(i))+beta*p(i));
    denom=N+sum(exp(alpha*log(f)+beta*p));
    ms=num/denom;
    g=C(i)*f(i)-min(ms*M,f(i)*S(i))*p(i);
