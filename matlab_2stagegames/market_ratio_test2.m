%create coefficients, US and WN in LAS PHX
markets=4;
r = 0:.1:1;
cd('C:/Users/Reed/Desktop/vaze_competition_paper/matlab_2stagegames')
M=cell(markets,1);
for i =1:markets
    M{i}=csvread(sprintf('r_mod%d.csv',i),1,1);
end

%optimization options
options = optimset('Display', 'off') ;

fs=zeros(numel(r),markets);

for row_ind=1:numel(r)
    fprime=cell(markets,1);
    for i =1:markets
        fprime{i}=M{i}(row_ind,1);
    end
    freq_ind=cell(markets,1);
    for i =1:markets
        freq_ind{i}=M{i}(row_ind,2);
    end
    min_coefs=cell(markets,1);
    for i =1:markets
        min_coefs{i}=-M{i}(row_ind,5:10);
    end
    F=zeros(markets,1);
    for i =1:markets
        F(i)=M{i}(row_ind,3);
    end
    bf=zeros(markets,1);
    for i =1:markets
        bf(i)=M{i}(row_ind,4);
    end
    
    %set up profit function for this carrier
    f=ones(markets,1)';
    profit_func=@(f)profit_minigame(f,markets,min_coefs, fprime,freq_ind);
    %display(min_coefs)
    %optimize frequencies of this carrier for profit
    [x, profit]=fmincon(profit_func,f,bf',18*sum(F),[],[],zeros(markets,1),ones(markets,1)*inf,[],options);
    fs(row_ind,:)=x;    
end
A = zeros(numel(r),5);
A(:,1)=r;



    