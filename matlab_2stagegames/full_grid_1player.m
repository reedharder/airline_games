tic
options = optimset('Display', 'off') ;
fvals_1p=zeros(20,6);
count=0;
S=10000;
N=0.5; %The exponential of the utility of the no-fly option
alpha=1.29;
beta=-0.0045;
M=1000; C=10000;
p0=100;
eps=0.1;
c=0;
for f1=1:20   
    p1=p0;
    diff=(eps+1);
    f=f1;    
    fun=@(p1)profit_1player(p1,f1,M,S,C,alpha,beta,N);
    [x1, fval]=fmincon(fun,p0,[],[],[],[],0,[],[],options); 
    p1=x1;                
    count=count+1;
    fvals_1p(count,:)=[alpha,beta,N,f1,fval,p1];   
end
toc