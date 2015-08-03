%coef check
c=[229325.30312328768, 14434.064090708493, -29137.724723287673, -354.96380931506854, 1084.2409589041097, -496.8325686301371];
c2 = [229325.30312328768, -29137.724723287673, 15078.017400543913, 1084.2409589041097, -354.96380931506854, -496.8325686301371];

f = 1:.1:30;
f2=3.9;
freqs = [ones(numel(f),1),f',ones(numel(f),1)*f2,f'.^2,(ones(numel(f),1)*f2).^2,f'*f2]';
prof=c*freqs;
plot(f,prof)
f_i=1;
profit_func=@(f_i)profit_stage1_network(f_i,c,1,13,{[1,3.9]});
%optimize frequencies of this carrier for profit
%[x_i, profit]=fmincon(profit_func,f_i,carrier.A,carrier.b,[],[],zeros(numel(carrier.Markets),1),ones(numel(carrier.Markets),1)*inf,[],options);
[x_i, profit]=fmincon(profit_func,f_i,[],[],[],[],zeros(1,1),ones(1,1)*inf,[],options);



%1 market 1 carrier
%input: Cnew, Mnew, freq empiricial, subF, block hours
fs= zeros(10,1);
for i=1:10
    input = [886.337,834.0636,7.36,3.862,2.43]; %input for AS,SAN_SEA
    Cold = 10000;
    Cnew = input(1);
    Mold = 1000;
    Mnew = input(2); 
    empirical_freq=input(3);
    subF = input(4);
    base =[-95164.0447,-36238.3083,1148.0305];
    base(3) = base(3)+i*100;
    transcoef =[-(Mnew/Mold)*base(1),(Mnew/Mold)*(Cold-base(2))-Cnew,-(Mnew/Mold)*base(3) ];
    options = optimset('Display', 'off') ;
    f_i=ones(1,1);
    %set up profit function for this carrier
    profit_func=@(f_i)profit_minigame(f_i,transcoef,1);
    %optimize frequencies of this carrier for profit
    b=subF*18;
    a=2*(45/60 + input(5));
    [x_i, profit]=fmincon(profit_func,f_i,[],[],[],[],0,[],[],options);
    fs(i)=x_i;

end


%1 market 2 carrier
%input: Cnew, Mnew, freq empiricial, subF, block hours
fs= zeros(10,2);
%%AS-> Qx, AA  - > MQ,  OO divide up
for i=1:10
    input1 = [267.119,2428.223,6.72,1.85,.924]; %input for AA,LAX_SFO
    input2 = [334.337,2428.223,17.04,3.22,.929]; %input for UA,LAX_SFO
    Cold = 10000;
    Cnew = input1(1);
    Mold = 1000;
    Mnew = input1(2); 
    empirical_freq1=input1(3);
    subF1 = input1(4);
    base =[-274960.0,-16470.0,34936.0,425.6,-1300.0,595.7];
    base(4) = base(4)+i*100;
    b1=subF1*18;
    a1=2*(45/60 + input1(5));
    transcoef1 =  [-(Mnew/Mold)*base(1),(Mnew/Mold)*(Cold-base(2))-Cnew, -(Mnew/Mold)*base(3), -(Mnew/Mold)*base(4), -(Mnew/Mold)*base(5), -(Mnew/Mold)*base(6)];
    Cold = 10000;
    Cnew = input2(1);
    Mold = 1000;
    Mnew = input2(2); 
    empirical_freq1=input2(3);
    subF2 = input2(4);    
    b2=subF2*18;
    a2=2*(45/60 + input2(5));
    transcoef2 =  [-(Mnew/Mold)*base(1),-(Mnew/Mold)*base(2),(Mnew/Mold)*(Cold-base(3))-Cnew,  -(Mnew/Mold)*base(4), -(Mnew/Mold)*base(5), -(Mnew/Mold)*base(6)];
    options = optimset('Display', 'off') ;
    f_i=ones(2,1);
    eps=0.001;    
    f1_0=1;
    f2_0=1;   
    diff1=eps+1;
    diff2=eps+1;    
    f1=f1_0; f2=f2_0;
    while (diff1>eps || diff2>eps)
        
        fun1=@(f1)profit_minigame([f1, f2],transcoef1,2);
        [x1, fval1]=fmincon(fun1,f1,a1,b1,[],[],0,inf, [],options);
        diff1=abs(f1-x1);
        f1=x1;
        fun2=@(f2)profit_minigame([f1, f2],transcoef2,2);
        [x2, fval2]=fmincon(fun2,f2,a2,b2,[],[],0,inf,[],options);
        diff2=abs(f2-x2);
        f2=x2;       
    end
    fs(i,:)=[x1,x2];
end

 