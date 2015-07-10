function g=create_A_b_matrices(profits,alpha,beta,f_other1,f_other2)
A=zeros(20,2);
b=zeros(20,1);
f=profits(:,6);
for i=1:19
    ind1=(i-1)*400+(f_other1-1)*20+f_other2
    ind2=(i-1)*400+(f_other1-1)*20+400+f_other2
    y0=f(round(ind1));
    y1=f(round(ind2));
    m=(y1-y0)/(i+1-i);
    b(i)=y0-m*i;
    A(i,1)=-m;
    A(i,2)=1;
end
A(20,1)=1; A(20,2)=0; b(20)=20;
g=[A b];