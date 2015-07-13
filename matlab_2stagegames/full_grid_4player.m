tic
options = optimset('Display', 'off') ;
ps=zeros(160000,10);
fvals=zeros(160000,10);
count=0;
S=ones(4,1)*10000;
%N=0.5; %The exponential of the utility of the no-fly option
N=0;
alpha=1.29;
beta=-0.0045;
M=1000; C=ones(4,1)*10000;
p0=100;
eps=0.1;
c=0;
for f1=1:20
    for f2=1:20
        for f3=1:20
            for f4=1:20
                p=ones(4,1)*p0;
                diff=ones(4,1)*(eps+1);
                f=[f1;f2;f3;f4];
                fval=zeros(4,1);
                while (diff(1)>eps || diff(2)>eps || diff(3)>eps || diff(4)>eps)
                    for i=1:4
                        p_i=p(i);
                        fun_i=@(p_i)profit_nplayer(p,f,M,S,C,alpha,beta,N,i,p_i);
                        [x_i, fval(i)]=fmincon(fun_i,p0,[],[],[],[],0,[],[],options);
                        diff(i)=abs(p(i)-x_i);
                        p(i)=x_i;
                    end
                    c=c+1;
                end
                count=count+1;
                ps(count,:)=[alpha,beta,f1,f2,f3,f4,p(1),p(2),p(3),p(4)];
                fvals(count,:)=[alpha,beta,f1,f2,f3,f4,fval(1),fval(2),fval(3),fval(4)];
                if (mod(count,10000)==0)
                    display(count)
                    toc
                end
            end
        end
    end
end
toc