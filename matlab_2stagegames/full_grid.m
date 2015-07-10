ps=zeros(1*400,6);
fvals=zeros(1*400,6);
count=0;
S1=10000; S2=10000;
N=0.5; %The exponential of the utility of the no-fly option
for alpha=1.29:0.04:1.29
    for beta=-0.0045:0.0005:-0.0045
        for f1=1:20
            for f2=1:20
                M=1000; C1=10000; C2=10000;
                p0=100;
                p1=p0; p2=p0;
                eps=0.1;
                diff1=eps+1;
                diff2=eps+1;
                c=0;
                while (diff1>eps || diff2>eps)
                    fun1=@(p1)profit1(p1,p2,f1,f2,M,S1,S2,C1,C2,alpha,beta,N);
                    [x1, fval1]=fmincon(fun1,p0,[],[],[],[],0);
                    diff1=abs(p1-x1);
                    p1=x1;
                    fun2=@(p2)profit2(p2,p1,f1,f2,M,S1,S2,C1,C2,alpha,beta,N);
                    [x2, fval2]=fmincon(fun2,p0,[],[],[],[],0);
                    diff2=abs(p2-x2);
                    p2=x2;
                    c=c+1;
                end
                count=count+1;
                ps(count,:)=[alpha,beta,f1,f2,p1,p2];
                fvals(count,:)=[alpha,beta,f1,f2,fval1,fval2];
            end
        end
    end
end
