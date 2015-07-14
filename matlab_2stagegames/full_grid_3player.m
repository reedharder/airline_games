tic
ps3=zeros(8000,8);
fvals3=zeros(8000,8);
options = optimset('Display', 'off') ;
count=0;
S=ones(3,1)*10000;
N=0; %The exponential of the utility of the no-fly option
alpha=1.29;
beta=-0.0045;
M=1000; C=ones(3,1)*10000;
p0=100;
eps=0.1;
c=0;
for f1=1:20
    for f2=1:20
        for f3=1:20
            p=ones(3,1)*p0;
            diff=ones(3,1)*(eps+1);
            f=[f1;f2;f3];
            fval=zeros(3,1);
            while (diff(1)>eps || diff(2)>eps || diff(3)>eps)
                for i=1:3
                    p_i=p(i);
                    fun_i=@(p_i)profit_nplayer(p,f,M,S,C,alpha,beta,N,i,p_i);
                    [x_i, fval(i)]=fmincon(fun_i,p0,[],[],[],[],0,[],[],options);
                    diff(i)=abs(p(i)-x_i);
                    p(i)=x_i;
                end
                c=c+1;
            end
            count=count+1;
            ps3(count,:)=[alpha,beta,f1,f2,f3,p(1),p(2),p(3)];
            fvals3(count,:)=[alpha,beta,f1,f2,f3,fval(1),fval(2),fval(3)];
            if (mod(count,1000)==0)
                display(count)
                toc
            end
        end
    end
end
