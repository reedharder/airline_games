alpha=fvals(:,1);
beta=fvals(:,2);
f1=fvals(:,3);
f2=fvals(:,4);
f3=fvals(:,5);
%X=[alpha,beta,f1,f2, ...
%    beta.^2,f1.^2,f2.^2,f1.*f2,alpha.*beta,alpha.*f1,alpha.*f2,beta.*f1,beta.*f2, ...
%    beta.*f1 .^2,alpha.*f2.^2,beta.*f2.^2,alpha.*f1.*f2,beta.*f1.*f2];
%X=[alpha,f1,f2, ...
%    f1.^2,f2.^2,f1.*f2,alpha.*f1,alpha.*f2,alpha.*f1.*f2];
X=[f1,f2,f3,f1.^2,f2.^2,f3.^2,f1.*f2,f1.*f3,f2.*f3];
%X=[f1,f2,f1.^2,f2.^2,f1.*f2,f1.^3,f2.^3,f1.*f1.*f2,f1.*f2.*f2,...
%    f1.^4,f2.^4,f1.*f1.*f1.*f2,f1.*f2.*f2.*f2,f1.*f1.*f2.*f2];
%X=[f1,f2];
%X=[f1,f2,f1.^2,f2.^2,f1.*f2];
Y=fvals(:,6);

%X=X(abs(beta+0.005)<0.00001,:);
%Y=Y(abs(beta+0.005)<0.00001,:);
mdl=fitlm(X,Y)
%for beta=-0.005, parameter estimates are:
%-2.8166e+05, 17050, 6363.4, 31351, 389.95, -1336.4, -479.07, -15814,
%2747.8, 792.22
coef=mdl.Coefficients.Estimate;