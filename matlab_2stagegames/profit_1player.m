function g=profit_1player(p1,f1,M,S,C,alpha,beta,N)
    num=exp(alpha*log(f1)+beta*p1);
    denom=exp(alpha*log(f1)+beta*p1)+N;
    ms=num/denom;
    g=C*f1-min(ms*M,f1*S)*p1;
