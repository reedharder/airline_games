function g=create_A_b_matrices(profits,alpha,beta,f_other)
f=profits(abs(alpha-profits(:,1))<0.00001,:);
size(f)
f=profits(abs(beta-f(:,2))<0.00001,5);
size(f)
A=zeros(20,2);
b=zeros(20,1);
for i=1:19
    y0=f(round((i-1)*20+f_other));
    y1=f(round((i+1-1)*20+f_other));
    m=(y1-y0)/(i+1-i);
    b(i)=y0-m*i;
    A(i,1)=-m;
    A(i,2)=1;
end
A(20,1)=1; A(20,2)=0; b(20)=20;
g=[A b];