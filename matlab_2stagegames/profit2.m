function g=profit2(p2,p1,f1,f2,M,S1,S2,C1,C2,alpha,beta,N)
    num=exp(alpha*log(f2)+beta*p2);
    denom=exp(alpha*log(f1)+beta*p1)+exp(alpha*log(f2)+beta*p2)+N;
    ms=num/denom;
    g=C2*f2-min(ms*M,f2*S2)*p2;
