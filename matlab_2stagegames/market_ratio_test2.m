%for two competitor markets only currently
%create coefficients, US and WN in LAS PHX
markets=4;
market_of_interest = 'WN';
named_markets={'LAS_PHX','LAS_SAN','LAS_SJC','ONT_PHX'};
num_modifications = 11; %how many different versions of particular coefficient being tested, independent of r
num_modifications_quad = 11;
%labels: interaction, quadratic, r
row_labels=csvread('term_mods.txt');
r = 0:.1:1;
%labels for coefficients
%%modification_labels=reshape(repmat(linspace(-1,1,21),numel(r),1),[numel(r)*num_modifications,1]);

cd('C:/Users/Reed/Desktop/vaze_competition_paper/matlab_2stagegames')
M=cell(markets,1);
for i =1:markets
    M{i}=csvread(sprintf('r_mod_%s.csv',named_markets{i}),1,1);
end

%optimization options
options = optimset('Display', 'off') ;
%intialize estimated frequencies
fs=zeros(numel(r)*num_modifications*num_modifications_quad,markets);
%intialize empirical frequencies
true_fs=zeros(numel(r)*num_modifications*num_modifications_quad,markets);

for row_ind=1:numel(r)*num_modifications*num_modifications_quad
    %obviously below could be made more efficient
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
    market_fs=zeros(markets,1);
    for i =1:markets
        market_fs(i)=M{i}(row_ind,12);
    end
    for i=1:numel(r)*num_modifications*num_modifications_quad
        true_fs(i,:)=market_fs;
    end
    %set up profit function for this carrier
    f=ones(markets,1)';
    profit_func=@(f)profit_minigame(f,markets,min_coefs, fprime,freq_ind);
    %display(min_coefs)
    %optimize frequencies of this carrier for profit
    [x, profit]=fmincon(profit_func,f,bf',18*sum(F),[],[],zeros(markets,1),ones(markets,1)*inf,[],options);
    fs(row_ind,:)=x; 
    
end
%construct output table
A = zeros(numel(r)*num_modifications*num_modifications_quad,markets+3);
A(:,1)=repmat(r',num_modifications,1);
A(:,2:markets+1)=fs;
%MAPE
MAPE = sum(abs(fs - true_fs),2)./sum(true_fs,2);
A(:,markets+2)=MAPE;
%R^2 analogy
R2 = 1-sum((fs-true_fs).^2,2)./sum((true_fs-repmat(mean(true_fs,2),1,markets)).^2,2);
A(:,markets+3)=R2;
B=horzcat(modification_labels,A);







    