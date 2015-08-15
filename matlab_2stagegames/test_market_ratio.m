%create coefficients, US and WN in LAS PHX
%DAILY_FREQ	FLIGHT_COST	MARKET_TOT	MARKET_COMPETITORS	MARKET_RANK	MS_TOT	AOTP_FLIGHT_TIME	BACKFOURTH
US_LAS_PHX = [13.14794521,	2741.966489,	2893.984932,	2,	2,0.458567582,	0.753976102,	3.007952204];
WN_LAS_PHX=[18.09863014,	2471.672902,	2893.984932,	2,	1,	0.541432418,	0.755375861,	3.010751722];
%for WN...
Cold=10000;
Mold = 1000;
r = .2:.1:9;
cd('C:/Users/Reed/Desktop/vaze_competition_paper/matlab_2stagegames')
M1 = csvread('r_mod1.csv',1,1);
M2 = csvread('r_mod2.csv',1,1);
%optimization options
options = optimset('Display', 'off') ;
M_size =size(M);
fs = zeros(M_size(1),2);
for row_ind=1:M_size(1)
    min_coefs1 = -M1(row_ind,5:10);
    fprime1 = M1(row_ind,1);
    freq_ind1 = M1(row_ind,2);
    min_coefs2 = -M2(row_ind,5:10);
    fprime2 = M2(row_ind,1);
    freq_ind2 = M2(row_ind,2);
    F =  [M1(row_ind,3),M2(row_ind,3)];
    bf =  [M1(row_ind,4),M2(row_ind,4)];
    %set up profit function for this carrier
    f=[1,1];
    profit_func=@(f)profit_minigame(f,min_coefs1, fprime1,freq_ind1,min_coefs2, fprime2,freq_ind2);
    %display(min_coefs)
    %optimize frequencies of this carrier for profit
    [x, profit]=fmincon(profit_func,f,bf,18*sum(F),[],[],[0,0],[inf,inf],[],options);
    fs(row_ind,:)=x;    
end
    