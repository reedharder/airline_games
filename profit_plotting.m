m=1;
n=20;
f1=m:n;
f2=m:n;
corig = -[ -274960.0	-16470.0	34936.0	425.6	-1300.0	595.7];
cc = [corig corig];
%cc=[47117145.6	4538028.174081166	-5986632.960000001	-72930.816	222768.00000000003	-102079.15200000002	];
Z=zeros(numel(f1),numel(f2));
for i=1:numel(f1)
    for j=1:numel(f2)
        Z(i,j)=cc(1)+cc(2)*f1(i) + cc(3)*f2(j) + cc(4)*f1(i)^2 +cc(5)+f2(j)^2 +cc(6)*f1(i)*f1(j);
    end
end
surf(f1,f2,Z)
rotate3d on






