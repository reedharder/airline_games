cd('C:/Users/Reed/Desktop/vaze_competition_paper')
fid = fopen('carrier_data.txt','r');
%optimization options
options = optimset('Display', 'off') ;
tic
%get number of carriers and markets
tline = fgetl(fid);
first_line = strsplit(tline);
num_carriers = str2double(first_line{1});
num_segments = str2double(first_line{2});
%get number of competitors in each market
tline = fgetl(fid);
second_line = strsplit(tline);
segment_competitors = eval(second_line{1});
%get data for each carrier
tline = fgetl(fid);
carriers = {};
line = 1;
while ischar(tline)
  carrier_cell = strsplit(tline);  
  carrier.num = line;
  carrier.A = eval(carrier_cell{1});
  carrier.b = eval(carrier_cell{2})';
  carrier.Markets = eval(carrier_cell{3});
  carrier.carrier_freq_ind = eval(carrier_cell{4});
  carrier.coef = eval(carrier_cell{5});
  carrier.freqs = zeros(numel(carrier.Markets),1);
  carrier.profits = zeros(numel(carrier.Markets),1);
  carriers{line} = carrier;
  tline = fgetl(fid);
  line = line+1;
end
fclose(fid);

eps=0.001;
diffs = ones(num_carriers) +eps;
%initialize markets array, constaining frequencies of carriers competing in
%that market
Market_freqs = {};
for i=1:num_segments
    Market_freqs{i}=ones(segment_competitors(i));
end
%myopic best response: each carrier decides frequencies for all the
%segments that it is competing on
loop=1;
while (sum(diffs>eps)>0)
    for carrier_ind=1:num_carriers
        carrier = carriers{carrier_ind};
        %current frequencies of carrier on all of its market segments
        current_markets = Market_freqs(carrier.Markets);
        f_i = zeros(numel(carrier.Markets),1);
        for i=1:numel(carrier.Markets)
            current_market_freqs = current_markets(i);
            %get frequency of current carrier curresponding to current
            %market
            f_i(i) = current_market_freqs(carrier.freq_inds(i));           
        end
        %set up profit function for this carrier
        profit_func=@(f_i)profit_stage1_network(f_i,carrier.coef, carrier.freq_inds,carrier.Markets,current_markets);
        %optimize frequencies of this carrier for profit
        [x_i, profit]=fmincon(profit_func,f_i,carrier.A,carrier.b,[],[],0,inf,[],options);
        %check for convergence
        diffs(carrier_ind)=sum(abs(f_i-x_i));
        %set new optimal frequencies into market frequencies data structure
        for i=1:numel(carrier.Markets)
            %frequency in market i 
            new_market_freq = x_i(i);
            %get current market frequencies
            current_market_freqs = current_markets(i);
            %update frequency of current carrier in current market
            current_market_freqs(carrier.freq_inds(i))= new_market_freq;
            %put new frequencies in market back into book keeping  market
            %frequencies data structure
            Market_freqs{carrier.Markets(i)}=current_market_freqs;                  
        end
        %save current frequencies and profits for each market for this
        %carrier
        carrier.freqs = x_i;
        carrier.profits = profits;
    end
    display(loop)
    loop =  loop +1;
    toc
end
%final frequencies will be contained in Market freqs and carrier.freqs
%(organized by market and carrier, respectively)
%corresponding profits are in carrier.profits

