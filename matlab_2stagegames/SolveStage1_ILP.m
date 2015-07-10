C1=10000;
C2=10000;
alpha=1.32;
beta=-0.005;
log_=[];
C0=10000;

f1_0=1;
f2_0=1;

eps=0.001;
diff1=eps+1;
diff2=eps+1;
c=0;
f1=f1_0; f2=f2_0;

profits=fvals;
profits(:,5:6)=-profits(:,5:6);

while (diff1>eps || diff2>eps)
    log_=[log_; f1,f2];
    Ab=create_A_b_matrices(profits,alpha,beta,f2);
    A=Ab(:,1:2); b=Ab(:,3);
    [x1, fval1]=intlinprog([(C1-C0);-1],[1],A,b,[],[],[1,-Inf],[20,Inf]);
    diff1=abs(f1-x1(1));
    f1=x1(1);
    Ab=create_A_b_matrices(profits,alpha,beta,f1);
    A=Ab(:,1:2); b=Ab(:,3);
    [x2, fval2]=intlinprog([(C2-C0);-1],[1],A,b,[],[],[1,-Inf],[20,Inf]);
    diff2=abs(f2-x2(1));
    f2=x2(1);
    c=c+1;
end

[f1,f2]
[fval1, fval2]