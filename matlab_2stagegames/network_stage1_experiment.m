%individual coeff, with bounds
cd('C:/Users/Reed/Desktop/vaze_competition_paper/matlab_2stagegames')
tic

%carrier fixing 
%['AA','AS','MQ','OO','QX','UA','US','WN']
fixed_carrier_mat = [1   0    1    1    1    1    1    1;
    1   1    1    1    1    0    1    1;
    1   1    1    1    1    1    0    1;
    1   1    1    1    1    1    1    0;];
for fixed_carrier_ind=1:4
    for coef_ind = [2,4,6,15,17,19,24,26,28,37,39,41]
        for modif =-1:.1:1.5%[-.5 -.4 -.3 -.2 -.1  0 .1 .2 .3 .4 .5]

            modstr = num2str(modif,'%.1f');
            progress = sprintf('%d_%s',coef_ind,modstr);
            display(progress)
            fn_open = sprintf('carrier_data_WN_%d_%s.txt',coef_ind,modstr);

            fid = fopen(fn_open,'r');
            %carrier fixing 
                        %['AA','AS','MQ','OO','QX','UA','US','WN']
            %%fixed_carrier = [1   0    1    1    1    0    0    0];
            fixed_carrier = fixed_carrier_mat(fixed_carrier_ind,:);
            %optimization options
            options = optimset('Display', 'off') ;

            %get number of carriers and markets
            tline = fgetl(fid);
            first_line = strsplit(tline);
            num_carriers = str2double(first_line{1});
            num_segments = str2double(first_line{2});
            %get number of competitors in each market
            tline = fgetl(fid);
            second_line = strsplit(tline);
            segment_competitors = eval(second_line{1});
            %get empirical frequencies of each competitor in each market (markets in
            %order of segment_competitors above, competitors within markets in order of
            %market rank)
            tline = fgetl(fid);
            third_line = strsplit(tline);
            empirical_freqs = eval(third_line{1});
            %get carrier index of all frequencies above in same sized array
            tline = fgetl(fid);
            fourth_line = strsplit(tline);
            empirical_carriers = eval(fourth_line{1});
            %get data for each carrier
            tline = fgetl(fid);
            carriers = {};
            line = 1;
            %keep track of which carrier we are on, lock frequencies 
            while ischar(tline)
              carrier_cell = strsplit(tline);  
              carrier.num = line;
              carrier.A = eval(carrier_cell{1});
              carrier.b = eval(carrier_cell{2})';
              carrier.Markets = eval(carrier_cell{3});
              carrier.freq_inds = eval(carrier_cell{4});
              carrier.coef = eval(carrier_cell{5});
              carrier.freqs = zeros(numel(carrier.Markets),1);
              carrier.profits = zeros(numel(carrier.Markets),1);
              carriers{line} = carrier;
              tline = fgetl(fid);
              line = line+1;
            end
            fclose(fid);

            eps=0.1;
            diffs = ones(num_carriers,1) +eps;
            %initialize markets array, constaining frequencies of carriers competing in
            %that market
            Market_freqs = {};
            for i=1:num_segments
                Market_freqs{i}=ones(segment_competitors(i),1);
            end
            %input fixed empirical frequencies into array above, containing empirical market
            %frequencies
            empirical_freq_index = 1;
            for i=1:num_segments
                for j=1:segment_competitors(i)
                    %if current carrier as marked by empirical_carrier is marked by
                    %used as fixed, set initial frequency in Market_freqs data
                    %structure to be the empirical frequency
                    if (fixed_carrier(empirical_carriers(empirical_freq_index)) == 1)
                        Market_freqs{i}(j)=empirical_freqs(empirical_freq_index);            
                    end
                    %increment empirical frequency index
                    empirical_freq_index = empirical_freq_index + 1; 
                end
            end
            %myopic best response: each carrier decides frequencies for all the
            %segments that it is competing on
            loop=1;
            while (sum(diffs(fixed_carrier == 0)>eps)>0) %loop until converges (just for non fixed diff indices)
                for carrier_ind=1:num_carriers
                    %if carrier is not fixed, run optimization 
                    %(MAY WANT TO COMPUTE PROFIT EVENTUALLY FOR fixed_carrier(carrier_ind) ==1 FOR COMPARISON 
                    if (fixed_carrier(carrier_ind) ==0)           
                        carrier = carriers{carrier_ind};
                        %current frequencies of carrier on all of its market segments
                        current_markets = Market_freqs(carrier.Markets);
                        f_i = zeros(numel(carrier.Markets),1);
                        %loop through markets, get frequency of carrier at these markets
                        for i=1:numel(carrier.Markets)
                            current_market_freqs = current_markets{i};
                            %get frequency of current carrier corresponding to current
                            %market
                            f_i(i) = current_market_freqs(carrier.freq_inds(i));      
                        end

                        %set up profit function for this carrier
                        profit_func=@(f_i)profit_stage1_network(f_i,carrier.coef, carrier.freq_inds,carrier.Markets,current_markets);
                        %optimize frequencies of this carrier for profit
                        [x_i, profit]=fmincon(profit_func,f_i,carrier.A,carrier.b,[],[],zeros(numel(carrier.Markets),1),ones(numel(carrier.Markets),1)*inf,[],options);
                        %[x_i, profit]=fmincon(profit_func,f_i,[],[],[],[],zeros(numel(carrier.Markets),1),ones(numel(carrier.Markets),1)*inf,[],options);
                        %%%%sprintf('carrier: %d',carrier_ind)
                        %%%%display(x_i)
                        %check for convergence
                        diffs(carrier_ind)=sum(abs(f_i-x_i));
                        %set new optimal frequencies into market frequencies data structure
                        for i=1:numel(carrier.Markets)
                            %frequency in market i 
                            new_market_freq = x_i(i);
                            %get current market frequencies
                            current_market_freqs = current_markets{i};
                            %update frequency of current carrier in current market
                            current_market_freqs(carrier.freq_inds(i))= new_market_freq;
                            %put new frequencies in market back into book keeping  market
                            %frequencies data structure
                            Market_freqs{carrier.Markets(i)}=current_market_freqs;                  
                        end

                        %save current frequencies and profits for each market for this
                        %carrier
                        carrier.freqs = x_i;
                        carrier.profits = profit;
                    end
                end
                %display loop number and time
                %%%display(loop)
                loop =  loop +1;
                %%%toc
            end
            %final frequencies will be contained in Market freqs and carrier.freqs
            %(organized by market and carrier, respectively)
            %corresponding profits are in carrier.profits

            %write market freqs into a reasonable matrix, columns as market, rank, and
            %frequency, to be matched with sorted dataframe t100ranked in python (see
            %function FUNCTION NAME
            freq_results_mat = zeros(sum(segment_competitors),3);
            %row index
            row_ind =1;
            %for each market...
            for mk=1:numel(segment_competitors)
                market_data = Market_freqs{mk};
                %for each competitor in market...
                rank=1;
                for row=row_ind:row_ind+segment_competitors(mk) - 1
                    freq_results_mat(row,:)=[mk,rank,market_data(rank)];
                    rank=rank+1;
                end
                row_ind = row_ind+segment_competitors(mk);
            end
            fn_out = sprintf('exp_results_fixed_WNsep%d_%d_%s.txt',fixed_carrier_ind,coef_ind,modstr);
            dlmwrite(fn_out,freq_results_mat,'delimiter',',','precision','%.4f')
            display(loop)
        end
    end
    
end
toc
display('PART1 DONE')



%all fixed carriers, no bounds
%carrier fixing 
%['AA','AS','MQ','OO','QX','UA','US','WN']
fixed_carrier_mat = [1   0    1    1    1    1    1    1;
    1   1    1    1    1    0    1    1;
    1   1    1    1    1    1    0    1;
    1   1    1    1    1    1    1    0;];
for fixed_carrier_ind=4 %1:4
    for coef_ind = [28,37,39,41]%[2,4,6,15,17,19,24,26,28,37,39,41]
        for modif =-1:.1:1

            modstr = num2str(modif,'%.1f');
            progress = sprintf('%d_%s',coef_ind,modstr);
            display(progress)
            fn_open = sprintf('carrier_data_WN_%d_%s.txt',coef_ind,modstr);

            fid = fopen(fn_open,'r');
            %carrier fixing 
                        %['AA','AS','MQ','OO','QX','UA','US','WN']
            %%fixed_carrier = [1   0    1    1    1    0    0    0];
            fixed_carrier = fixed_carrier_mat(fixed_carrier_ind,:);
            %optimization options
            options = optimset('Display', 'off') ;

            %get number of carriers and markets
            tline = fgetl(fid);
            first_line = strsplit(tline);
            num_carriers = str2double(first_line{1});
            num_segments = str2double(first_line{2});
            %get number of competitors in each market
            tline = fgetl(fid);
            second_line = strsplit(tline);
            segment_competitors = eval(second_line{1});
            %get empirical frequencies of each competitor in each market (markets in
            %order of segment_competitors above, competitors within markets in order of
            %market rank)
            tline = fgetl(fid);
            third_line = strsplit(tline);
            empirical_freqs = eval(third_line{1});
            %get carrier index of all frequencies above in same sized array
            tline = fgetl(fid);
            fourth_line = strsplit(tline);
            empirical_carriers = eval(fourth_line{1});
            %get data for each carrier
            tline = fgetl(fid);
            carriers = {};
            line = 1;
            %keep track of which carrier we are on, lock frequencies 
            while ischar(tline)
              carrier_cell = strsplit(tline);  
              carrier.num = line;
              carrier.A = eval(carrier_cell{1});
              carrier.b = eval(carrier_cell{2})';
              carrier.Markets = eval(carrier_cell{3});
              carrier.freq_inds = eval(carrier_cell{4});
              carrier.coef = eval(carrier_cell{5});
              carrier.freqs = zeros(numel(carrier.Markets),1);
              carrier.profits = zeros(numel(carrier.Markets),1);
              carriers{line} = carrier;
              tline = fgetl(fid);
              line = line+1;
            end
            fclose(fid);

            eps=0.1;
            diffs = ones(num_carriers,1) +eps;
            %initialize markets array, constaining frequencies of carriers competing in
            %that market
            Market_freqs = {};
            for i=1:num_segments
                Market_freqs{i}=ones(segment_competitors(i),1);
            end
            %input fixed empirical frequencies into array above, containing empirical market
            %frequencies
            empirical_freq_index = 1;
            for i=1:num_segments
                for j=1:segment_competitors(i)
                    %if current carrier as marked by empirical_carrier is marked by
                    %used as fixed, set initial frequency in Market_freqs data
                    %structure to be the empirical frequency
                    if (fixed_carrier(empirical_carriers(empirical_freq_index)) == 1)
                        Market_freqs{i}(j)=empirical_freqs(empirical_freq_index);            
                    end
                    %increment empirical frequency index
                    empirical_freq_index = empirical_freq_index + 1; 
                end
            end
            %myopic best response: each carrier decides frequencies for all the
            %segments that it is competing on
            loop=1;
            while (sum(diffs(fixed_carrier == 0)>eps)>0) %loop until converges (just for non fixed diff indices)
                for carrier_ind=1:num_carriers
                    %if carrier is not fixed, run optimization 
                    %(MAY WANT TO COMPUTE PROFIT EVENTUALLY FOR fixed_carrier(carrier_ind) ==1 FOR COMPARISON 
                    if (fixed_carrier(carrier_ind) ==0)           
                        carrier = carriers{carrier_ind};
                        %current frequencies of carrier on all of its market segments
                        current_markets = Market_freqs(carrier.Markets);
                        f_i = zeros(numel(carrier.Markets),1);
                        %loop through markets, get frequency of carrier at these markets
                        for i=1:numel(carrier.Markets)
                            current_market_freqs = current_markets{i};
                            %get frequency of current carrier corresponding to current
                            %market
                            f_i(i) = current_market_freqs(carrier.freq_inds(i));      
                        end

                        %set up profit function for this carrier
                        profit_func=@(f_i)profit_stage1_network(f_i,carrier.coef, carrier.freq_inds,carrier.Markets,current_markets);
                        %optimize frequencies of this carrier for profit
                        %[x_i, profit]=fmincon(profit_func,f_i,carrier.A,carrier.b,[],[],zeros(numel(carrier.Markets),1),ones(numel(carrier.Markets),1)*inf,[],options);
                        [x_i, profit]=fmincon(profit_func,f_i,[],[],[],[],zeros(numel(carrier.Markets),1),ones(numel(carrier.Markets),1)*inf,[],options);
                        %%%%sprintf('carrier: %d',carrier_ind)
                        %%%%display(x_i)
                        %check for convergence
                        diffs(carrier_ind)=sum(abs(f_i-x_i));
                        %set new optimal frequencies into market frequencies data structure
                        for i=1:numel(carrier.Markets)
                            %frequency in market i 
                            new_market_freq = x_i(i);
                            %get current market frequencies
                            current_market_freqs = current_markets{i};
                            %update frequency of current carrier in current market
                            current_market_freqs(carrier.freq_inds(i))= new_market_freq;
                            %put new frequencies in market back into book keeping  market
                            %frequencies data structure
                            Market_freqs{carrier.Markets(i)}=current_market_freqs;                  
                        end

                        %save current frequencies and profits for each market for this
                        %carrier
                        carrier.freqs = x_i;
                        carrier.profits = profit;
                    end
                end
                %display loop number and time
                %%%display(loop)
                loop =  loop +1;
                %%%toc
            end
            %final frequencies will be contained in Market freqs and carrier.freqs
            %(organized by market and carrier, respectively)
            %corresponding profits are in carrier.profits

            %write market freqs into a reasonable matrix, columns as market, rank, and
            %frequency, to be matched with sorted dataframe t100ranked in python (see
            %function FUNCTION NAME
            freq_results_mat = zeros(sum(segment_competitors),3);
            %row index
            row_ind =1;
            %for each market...
            for mk=1:numel(segment_competitors)
                market_data = Market_freqs{mk};
                %for each competitor in market...
                rank=1;
                for row=row_ind:row_ind+segment_competitors(mk) - 1
                    freq_results_mat(row,:)=[mk,rank,market_data(rank)];
                    rank=rank+1;
                end
                row_ind = row_ind+segment_competitors(mk);
            end
            fn_out = sprintf('exp_results_fixed_nobounds_WNsep%d_%d_%s.txt',fixed_carrier_ind,coef_ind,modstr);
            dlmwrite(fn_out,freq_results_mat,'delimiter',',','precision','%.4f')
            display(loop)
        end
    end
    
end
toc

display('PART2 DONE')

%all experiments, full competition, bounds

cd('C:/Users/Reed/Desktop/vaze_competition_paper/matlab_2stagegames')
tic
fns = textread('mkt_mod_fns.txt','%s');
%for coef_ind =[4, 6, 9, 11, 13, 15, 17, 19, 21, 22]%[2, 4, 6, 9, 11, 13, 15, 17, 19, 21, 22]
%    for modif =-.5:.1:.5 %.5:.1:1
for fn= 1:numel(fns)    
    %modstr = num2str(modif,'%.1f');
    %progress = sprintf('%d_%s',coef_ind,modstr);
    %display(progress)
    %%fn_open = sprintf('carrier_data_basemod_WNmod_MktDiv%d_%s.txt',coef_ind,modstr);
    fn_open = char(fns(fn));
    fid = fopen(fn_open,'r');
    display(fn_open)
    %carrier fixing 
                %['AA','AS','MQ','OO','QX','UA','US','WN']
    fixed_carrier = [1   0    1    1    1    0    0    0];

    %optimization options
    options = optimset('Display', 'off') ;

    %get number of carriers and markets
    tline = fgetl(fid);
    first_line = strsplit(tline);
    num_carriers = str2double(first_line{1});
    num_segments = str2double(first_line{2});
    %get number of competitors in each market
    tline = fgetl(fid);
    second_line = strsplit(tline);
    segment_competitors = eval(second_line{1});
    %get empirical frequencies of each competitor in each market (markets in
    %order of segment_competitors above, competitors within markets in order of
    %market rank)
    tline = fgetl(fid);
    third_line = strsplit(tline);
    empirical_freqs = eval(third_line{1});
    %get carrier index of all frequencies above in same sized array
    tline = fgetl(fid);
    fourth_line = strsplit(tline);
    empirical_carriers = eval(fourth_line{1});
    %get data for each carrier
    tline = fgetl(fid);
    carriers = {};
    line = 1;
    %keep track of which carrier we are on, lock frequencies 
    while ischar(tline)
      carrier_cell = strsplit(tline);  
      carrier.num = line;
      carrier.A = eval(carrier_cell{1});
      carrier.b = eval(carrier_cell{2})';
      carrier.Markets = eval(carrier_cell{3});
      carrier.freq_inds = eval(carrier_cell{4});
      carrier.coef = eval(carrier_cell{5});
      carrier.freqs = zeros(numel(carrier.Markets),1);
      carrier.profits = zeros(numel(carrier.Markets),1);
      carriers{line} = carrier;
      tline = fgetl(fid);
      line = line+1;
    end
    fclose(fid);

    eps=0.1;
    diffs = ones(num_carriers,1) +eps;
    %initialize markets array, constaining frequencies of carriers competing in
    %that market
    Market_freqs = {};
    for i=1:num_segments
        Market_freqs{i}=ones(segment_competitors(i),1);
    end
    %input fixed empirical frequencies into array above, containing empirical market
    %frequencies
    empirical_freq_index = 1;
    for i=1:num_segments
        for j=1:segment_competitors(i)
            %if current carrier as marked by empirical_carrier is marked by
            %used as fixed, set initial frequency in Market_freqs data
            %structure to be the empirical frequency
            if (fixed_carrier(empirical_carriers(empirical_freq_index)) == 1)
                Market_freqs{i}(j)=empirical_freqs(empirical_freq_index);            
            end
            %increment empirical frequency index
            empirical_freq_index = empirical_freq_index + 1; 
        end
    end
    %myopic best response: each carrier decides frequencies for all the
    %segments that it is competing on
    loop=1;
    while (sum(diffs(fixed_carrier == 0)>eps)>0) %loop until converges (just for non fixed diff indices)
        for carrier_ind=1:num_carriers
            %if carrier is not fixed, run optimization 
            %(MAY WANT TO COMPUTE PROFIT EVENTUALLY FOR fixed_carrier(carrier_ind) ==1 FOR COMPARISON 
            if (fixed_carrier(carrier_ind) ==0)           
                carrier = carriers{carrier_ind};
                %current frequencies of carrier on all of its market segments
                current_markets = Market_freqs(carrier.Markets);
                f_i = zeros(numel(carrier.Markets),1);
                %loop through markets, get frequency of carrier at these markets
                for i=1:numel(carrier.Markets)
                    current_market_freqs = current_markets{i};
                    %get frequency of current carrier corresponding to current
                    %market
                    f_i(i) = current_market_freqs(carrier.freq_inds(i));      
                end

                %set up profit function for this carrier
                profit_func=@(f_i)profit_stage1_network(f_i,carrier.coef, carrier.freq_inds,carrier.Markets,current_markets);
                %optimize frequencies of this carrier for profit
                [x_i, profit]=fmincon(profit_func,f_i,carrier.A,carrier.b,[],[],zeros(numel(carrier.Markets),1),ones(numel(carrier.Markets),1)*inf,[],options);
                %[x_i, profit]=fmincon(profit_func,f_i,[],[],[],[],zeros(numel(carrier.Markets),1),ones(numel(carrier.Markets),1)*inf,[],options);
                %%%%sprintf('carrier: %d',carrier_ind)
                %%%%display(x_i)
                %check for convergence
                diffs(carrier_ind)=sum(abs(f_i-x_i));
                %set new optimal frequencies into market frequencies data structure
                for i=1:numel(carrier.Markets)
                    %frequency in market i 
                    new_market_freq = x_i(i);
                    %get current market frequencies
                    current_market_freqs = current_markets{i};
                    %update frequency of current carrier in current market
                    current_market_freqs(carrier.freq_inds(i))= new_market_freq;
                    %put new frequencies in market back into book keeping  market
                    %frequencies data structure
                    Market_freqs{carrier.Markets(i)}=current_market_freqs;                  
                end

                %save current frequencies and profits for each market for this
                %carrier
                carrier.freqs = x_i;
                carrier.profits = profit;
            end
        end
        %display loop number and time
        %%%display(loop)
        loop =  loop +1;
        %%%toc
    end
    %final frequencies will be contained in Market freqs and carrier.freqs
    %(organized by market and carrier, respectively)
    %corresponding profits are in carrier.profits

    %write market freqs into a reasonable matrix, columns as market, rank, and
    %frequency, to be matched with sorted dataframe t100ranked in python (see
    %function FUNCTION NAME
    freq_results_mat = zeros(sum(segment_competitors),3);
    %row index
    row_ind =1;
    %for each market...
    for mk=1:numel(segment_competitors)
        market_data = Market_freqs{mk};
        %for each competitor in market...
        rank=1;
        for row=row_ind:row_ind+segment_competitors(mk) - 1
            freq_results_mat(row,:)=[mk,rank,market_data(rank)];
            rank=rank+1;
        end
        row_ind = row_ind+segment_competitors(mk);
    end
    %%fn_out = sprintf('exp_results_basemod_WNmod_MktDiv%d_%s.txt',coef_ind,modstr);
    fn_out = strcat('exp_results_500_',fn_open(13:numel(fn_open)));
    dlmwrite(fn_out,freq_results_mat,'delimiter',',','precision','%.4f')
    display(loop)

end

toc
display('PART3 DONE')

a=1.9;
c=20;
A=100;
alpha=.602;
gamma=.101;
for k=0:n-1
    ak=a/(k+1+A)^alpha;
    ck=c/(k+1)^gamma;
    delta=2*round(rand(p,1))-1;
    thetaplus=theta+ck*delta;
    thetaminus=theta-ck*delta;
    yplus=loss(thetaplus);
    yminus=loss(thetaminus);
    ghat=(yplus-yminus)./(2*ck*delta);
    theta=theta-ak*ghat;
end
theta