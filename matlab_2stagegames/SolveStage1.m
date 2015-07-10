coef1=[-2.8166e+05, 17050, 6363.4, 31351, 389.95, -1336.4, -479.07, -15814, 2747.8, 792.22];
coef2=coef1;
C1=5000;
C2=5000;
alpha=1.5;
log_=[];

f1_0=1;
f2_0=1;

eps=0.001;
diff1=eps+1;
diff2=eps+1;
c=0;
f1=f1_0; f2=f2_0;

while (diff1>eps || diff2>eps)
    log_=[log_; f1,f2];
    fun1=@(f1)profit1_stage1(coef1,alpha,f1,f2,C1,C2);
    [x1, fval1]=fmincon(fun1,f1,[],[],[],[],0);
    diff1=abs(f1-x1);
    f1=x1;
    fun2=@(f2)profit2_stage1(coef2,alpha,f1,f2,C1,C2);
    [x2, fval2]=fmincon(fun2,f2,[],[],[],[],0);
    diff2=abs(f2-x2);
    f2=x2;
    c=c+1;
end

[f1,f2]
[fval1, fval2]