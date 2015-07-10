function g=profit1(p1,p2,f1,f2,M,S1,S2,C1,C2,alpha,beta,N)
    num=exp(alpha*log(f1)+beta*p1);
    denom=exp(alpha*log(f1)+beta*p1)+exp(alpha*log(f2)+beta*p2)+N;
    ms=num/denom;
    g=C1*f1-min(ms*M,f1*S1)*p1;
