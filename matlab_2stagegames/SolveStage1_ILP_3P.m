C1=9000;
C2=9000;
C3=9000;
alpha=1.29;
beta=-0.0045;
log_=[];
C0=10000;

f1_0=1;
f2_0=1;
f3_0=1;

eps=0.1;
diff1=eps+1;
diff2=eps+1;
diff3=eps+1;
c=0;
f1=f1_0; f2=f2_0; f3=f3_0;

profits=fvals;
profits(:,6:8)=-profits(:,6:8);

while (diff1>eps || diff2>eps || diff3>eps)
    log_=[log_;f1,f2,f3];
    Ab=create_A_b_matrices_3P(fvals,alpha,beta,f2,f3);
    A=Ab(:,1:2); b=Ab(:,3);
    [x1, fval1]=intlinprog([(C1-C0);-1],[1],A,b,[],[],[1,-Inf],[20,Inf]);
    fval1
    diff1=abs(f1-x1(1));
    f1=x1(1);
    Ab=create_A_b_matrices_3P(fvals,alpha,beta,f1,f3);
    A=Ab(:,1:2); b=Ab(:,3);
    [x2, fval2]=intlinprog([(C2-C0);-1],[1],A,b,[],[],[1,-Inf],[20,Inf]);
    diff2=abs(f2-x2(1));
    f2=x2(1);
    Ab=create_A_b_matrices_3P(fvals,alpha,beta,f1,f2);
    A=Ab(:,1:2); b=Ab(:,3);
    [x3, fval3]=intlinprog([(C3-C0);-1],[1],A,b,[],[],[1,-Inf],[20,Inf]);
    diff3=abs(f3-x3(1));
    f3=x3(1);
    c=c+1;
end

[f1,f2,f3]
[fval1,fval2,fval3]