cd('C:/Users/Reed/Desktop/vaze_competition_paper/matlab_2stagegames')
%for two competitor markets only currently
%create coefficients, US and WN in LAS PHX
for markets=2:7
    market_of_interest = 'WN';
    %%2player markets
    %%named_markets={'LAS_PHX','LAS_SAN','LAS_SJC','LAS_SMF','OAK_PHX','OAK_SEA','ONT_PHX','PHX_SAN','PHX_SMF','SEA_SJC','SEA_SMF'};
    %%3player markets
    named_markets={'LAS_LAX','LAS_PDX','LAS_SEA','LAS_SFO','PDX_PHX','PHX_SEA','SAN_SFO'};
    num_modifications = 16; %how many different versions of particular coefficient being tested, independent of r
    num_modifications_quad = 16;
    %labels: interaction, quadratic, r
    row_labels=csvread('term_mods.txt');
    r = 0:.1:1;
    %labels for coefficients
    %%modification_labels=reshape(repmat(linspace(-1,1,21),numel(r),1),[numel(r)*num_modifications,1]);
    cd('C:/Users/Reed/Desktop/vaze_competition_paper/matlab_2stagegames')
    %read matrices and get number of competitors in each market
    M=cell(markets,1);
    numplayers  = zeros(markets,1);
    for i =1:markets
        M{i}=csvread(sprintf('r_mod_%s.csv',named_markets{i}),1,1);
        M_size =size(M{i});
        if (M_size(2) == 12)
            numplayers(i)=2;
        elseif (M_size(2) == 17)
            numplayers(i)=3;
        else
            error('M size is wrong')
        end
    end
    %optimization options
    options = optimset('Display', 'off') ;
    %intialize estimated frequencies
    fs=zeros(numel(r)*num_modifications*num_modifications_quad,markets);
    %get matrix of true frequencys for each market, a row for each test
    market_fs=zeros(markets,1);
    for i =1:markets
        if (numplayers(i)==2)
            market_fs(i)=M{i}(1,12);
        elseif (numplayers(i)==3)
            market_fs(i)=M{i}(1,17);
        end       
    end
    true_fs = repmat(market_fs',numel(r)*num_modifications*num_modifications_quad,1);


    for row_ind=1:numel(r)*num_modifications*num_modifications_quad
        
        if (mod(row_ind,100)==0)
            display(row_ind)
        end
        
        fprime=cell(markets,1);
        for i =1:markets
            if (numplayers(i)==2)
                fprime{i}=M{i}(row_ind,1);
            elseif (numplayers(i)==3)
                fprime{i}=M{i}(row_ind,1:2);
            end
        end
        freq_ind=cell(markets,1);
        for i =1:markets
            if (numplayers(i)==2)
                freq_ind{i}=M{i}(row_ind,2);
            elseif (numplayers(i)==3)
                freq_ind{i}=M{i}(row_ind,3);
            end
        end
        min_coefs=cell(markets,1);
        for i =1:markets
            if (numplayers(i)==2)
                min_coefs{i}=-M{i}(row_ind,5:10);
            elseif (numplayers(i)==3)
                min_coefs{i}=eval(M{i}(row_ind,6:15));
            end
        end
        F=zeros(markets,1);
        for i =1:markets
            if (numplayers(i)==2)
                F(i)=M{i}(row_ind,3);
            elseif (numplayers(i)==3)
                F(i)=M{i}(row_ind,4);
            end            
        end
        bf=zeros(markets,1);
        for i =1:markets
            if (numplayers(i)==2)
               bf(i)=M{i}(row_ind,4);
            elseif (numplayers(i)==3)
               bf(i)=M{i}(row_ind,5);
            end              
        end
        


        %set up profit function for this carrier
        f=ones(markets,1)';
        profit_func=@(f)profit_minigame(f,markets,min_coefs, fprime,freq_ind,numplayers);
        %display(min_coefs)
        %optimize frequencies of this carrier for profit
        [x, profit]=fmincon(profit_func,f,bf',18*sum(F),[],[],zeros(markets,1),ones(markets,1)*inf,[],options);
        fs(row_ind,:)=x; 

    end
    %construct output table
    A = zeros(numel(r)*num_modifications*num_modifications_quad,markets+2);
    %%A(:,1)=repmat(r',num_modifications,1);
    A(:,1:markets)=fs;
    %MAPE
    MAPE = sum(abs(fs - true_fs),2)./sum(true_fs,2);
    A(:,markets+1)=MAPE;
    %R^2 analogy
    R2 = 1-sum((fs-true_fs).^2,2)./sum((true_fs-repmat(mean(true_fs,2),1,markets)).^2,2);
    A(:,markets+2)=R2;
    B=horzcat(row_labels,A);
    dlmwrite(sprintf('market_output_%d_3player.csv',markets),B,'delimiter','\t','precision','%.2f')
end





    