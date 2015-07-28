%one player regression
f1=fvals_1p(:,4);
X1=[f1,f1.^2]; 
Y1=fvals_1p(:,5);
mdl1=fitlm(X1,Y1)
coef1=mdl1.Coefficients.Estimate;
display('1 player done')
%three player regression
alpha=fvals3(:,1);
beta=fvals3(:,2);
f1=fvals3(:,3);
f2=fvals3(:,4);
f3=fvals3(:,5);
X=[f1,f2,f3,f1.^2,f2.^2,f3.^2,f1.*f2,f1.*f3,f2.*f3];
Y=fvals3(:,6);
mdl3=fitlm(X,Y)
coef3=mdl3.Coefficients.Estimate;
display('3 player done')
% 4 player regression
f1=fvals(:,3);
f2=fvals(:,4);
f3=fvals(:,5);
f4=fvals(:,6);
X4=[f1,f2,f3,f4,f1.^2,f2.^2,f3.^2,f4.^2,f1.*f2,f1.*f3,f1.*f4,f2.*f3,f2.*f4,f3.*f4]; 
Y4=fvals(:,7);
mdl4=fitlm(X4,Y4)
coef4=mdl4.Coefficients.Estimate;
display('4 player done')


dlmwrite('coef_4p.csv',coef4','delimiter',',','precision','%.4f')
%X=[alpha,beta,f1,f2, ...
%    beta.^2,f1.^2,f2.^2,f1.*f2,alpha.*beta,alpha.*f1,alpha.*f2,beta.*f1,beta.*f2, ...
%    beta.*f1 .^2,alpha.*f2.^2,beta.*f2.^2,alpha.*f1.*f2,beta.*f1.*f2];
%X=[alpha,f1,f2, ...
%    f1.^2,f2.^2,f1.*f2,alpha.*f1,alpha.*f2,alpha.*f1.*f2];

%X=[f1,f2,f1.^2,f2.^2,f1.*f2,f1.^3,f2.^3,f1.*f1.*f2,f1.*f2.*f2,...
%    f1.^4,f2.^4,f1.*f1.*f1.*f2,f1.*f2.*f2.*f2,f1.*f1.*f2.*f2];
%X=[f1,f2];
%X=[f1,f2,f1.^2,f2.^2,f1.*f2];


%X=X(abs(beta+0.005)<0.00001,:);
%Y=Y(abs(beta+0.005)<0.00001,:);
%mdl=fitlm(X,Y)
%for beta=-0.005, parameter estimates are:
%-2.8166e+05, 17050, 6363.4, 31351, 389.95, -1336.4, -479.07, -15814,
%2747.8, 792.22
%coef=mdl.Coefficients.Estimate;