function g=profit1_stage1(coef,alpha,f1,f2,C1,C2)
    C0=10000;
    vars=[1,alpha,f1,f2,f1^2,f2^2,f1*f2,alpha*f1,alpha*f2,alpha*f1*f2];
    g=coef*vars'+(C1-C0)*f1;